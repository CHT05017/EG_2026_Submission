# cd Prompt
# bash train.sh

from __future__ import print_function, absolute_import
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'

import argparse
import os.path as osp
import sys
import os
import json
from pathlib import Path
from torch.backends import cudnn
import torch.nn as nn
import random
import torch
from tqdm import tqdm

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel
ROOT_PATH = Path(__file__).resolve().parents[2] # CLIPSR/

if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from tqdm import tqdm
import datetime
from dataset.get_dataloader import get_data
from loss.sup import SupConLoss
from model.make_model import make_clipsr

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

def main():
    args = parser.parse_args()

    os.makedirs(args.stage1_prompts_out_dir, exist_ok=True)

    if args.seed is not None:
        print("setting the seed to",args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    main_worker(args)

def _set_stage1_mode(model):
    base=model.module

    for m in [getattr(base, 'image_encoder', None),
              getattr(base, 'classifier', None),
              getattr(base, 'classifier_proj', None),
              getattr(base, 'bottleneck', None),
              getattr(base, 'bottleneck_proj', None)]:
        if m is not None:
            for p in m.parameters():
                p.requires_grad = False

    if hasattr(model.module if hasattr(model, 'module') else model, 'classifier'):
        classifier = getattr(model.module if hasattr(model, 'module') else model, 'classifier')
        if isinstance(classifier, torch.nn.Module):
            for p in classifier.parameters():
                p.requires_grad = False

    if hasattr(base, 'text_encoder'):
        for p in base.text_encoder.parameters():
            p.requires_grad = False

    if hasattr(base, 'prompt_learner'):
        for p in base.prompt_learner.parameters():
            p.requires_grad = True

    return base

from bisect import bisect_right

from torch.optim.lr_scheduler import *

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
    

import numpy as np
def warm_up_cosine_lr_scheduler(optimizer, epochs=100, warm_up_epochs=10, eta_min=1e-9):
    """
        Description:
            - Warm up cosin learning rate scheduler, first epoch lr is too small
        Arguments:
            - optimizer: input optimizer for the training
            - epochs: int, total epochs for your training, default is 100. NOTE: you should pass correct epochs for your training
            - warm_up_epochs: int, default is 5, which mean the lr will be warm up for 5 epochs. if warm_up_epochs=0, means no need
              to warn up, will be as cosine lr scheduler
            - eta_min: float, setup ConsinAnnealingLR eta_min while warm_up_epochs = 0
        Returns:
            - scheduler
    """
    if warm_up_epochs== 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    else:
        warm_up_with_cosine_lr = lambda epoch: eta_min + (epoch / warm_up_epochs) if epoch <warm_up_epochs else 0.5 * (
            np.cos((epoch - warm_up_epochs) / (epochs - warm_up_epochs) * np.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    return scheduler


def _make_optimizer_for_stage1(model):
    params = []
    keys = []

    for key, value in model.named_parameters():
        if 'prompt_learner' in key:
            params += [         # clipreid setting
                {
                    "params": [value],
                    "lr": 0.0035,
                    "weight_decay": 1e-4
                }
            ]
            keys += [keys]
    optimizer = torch.optim.Adam(params=params)
    return optimizer

def main_worker(args):

    stage1_loader,fake_mapping=get_data(args, args.dataset_name)

    supcon_loss = SupConLoss(device='cuda')

    print('='*80)
    print('Training the Prompts for {}'.format(args.dataset_name))
    print('='*80)

    model=make_clipsr(args.dataset_name)
    model.to('cuda')
    model=DataParallel(model)

    Stones=[30,100,150]
    model=_set_stage1_mode(model=model)
    optimizer_stage1=_make_optimizer_for_stage1(model=model)

    lr_scheduler=WarmupMultiStepLR(optimizer_stage1,Stones,gamma=0.1,warmup_factor=0.01,warmup_iters=10)
    
    epochs=args.epochs

    for epoch in tqdm(range(epochs)):
        train_prompt_epoch(args,model,stage1_loader,optimizer_stage1,supcon_loss,lr_scheduler,args.dataset_name,fake_mapping,epoch)
    prompt_path = osp.join(args.stage1_prompts_out_dir, f'{args.dataset_name}_prompt_{args.epochs}.pth')
    
    if True:
        prompt_learner = getattr(model, 'prompt_learner', None)
        if prompt_learner is None:
            raise RuntimeError('prompt learner is None!!!!!!!!!')
        
        state = {
            "state_dict": prompt_learner.state_dict(),
            'n_ctx': getattr(prompt_learner, 'n_cls_ctx', None),
            'dataset_name': args.dataset_name
        }

        torch.save(state, prompt_path)
        print(f'Prompt Weight for {args.dataset_name} saved at: {prompt_path}')

    print('='*80)
    print('Stage1 prompt training ends...Everything done...')
    print('='*80)


def train_prompt_epoch(args,model, loader, optimizer, supcon_loss, lr_scheduler=None, \
                        dataset_name=None,fake_mapping=None,epoch=None):
    model.train()
    losses = []

    loader_bar = tqdm(loader, desc="stage1 batches", leave=False)
    for iter, inputs in enumerate(loader_bar):  

        img_transformed, fpaths = inputs
        img_transformed = img_transformed.to('cuda')

        if fake_mapping is not None: 
            if dataset_name.lower() in ['bird','birds']:
                fake_key=[(item.split('/'))[-2] for item in fpaths]
            elif dataset_name.lower() in ['celeba','coco','div2k']:
                fake_key=[((osp.basename(fp)).split('.'))[0] for fp in fpaths]
            else:
                raise NotImplementedError('not supported yet...')
            
            fake_targets=torch.tensor([fake_mapping[key] for key in fake_key])
        else:
            raise NotImplementedError('U gave me an empty fake mapping... :(')    

        fake_targets = fake_targets.to('cuda')
        #print(fake_targets[:5])
        #print(fpaths[:5])
        #exit()
        

        optimizer.zero_grad()
        with torch.no_grad():
            img_features = model(x=img_transformed, get_image=True) # 512-D
            #print(img_features.shape) # (B,512)
            
        txt_features = model(index=fake_targets, get_text=True)
        #print(txt_features.shape) # (B,512)
        

        i2t = supcon_loss(img_features, txt_features, fake_targets,fake_targets)
        t2i = supcon_loss(txt_features, img_features, fake_targets, fake_targets)

        loss_stage1 = i2t + t2i

        loss_stage1.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        loader_bar.set_postfix(i2t=f"{i2t.item():.4f}", t2i=f"{t2i.item():.4f}", total_loss=f"{loss_stage1.item():.4f}")
        #if epoch % 50 == 0 or (epoch+1)==args.epochs:
        #    print(f'\n[stage1]: i2t loss: {i2t}\n[stage1]: t2i loss: {t2i}')
        #    print(f'[stage1]: total loss: {loss_stage1}')

def _make_optimizer_for_stage1(model):
    params = []
    keys = []

    for key, value in model.named_parameters():
        if 'prompt_learner' in key:
            params += [         # clipreid setting
                {
                    "params": [value],
                    "lr": 0.0035,
                    "weight_decay": 1e-4
                }
            ]
            keys += [keys]
    optimizer = torch.optim.Adam(params=params)
    return optimizer



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stage1 training")

    parser.add_argument('--stage1_prompts_out_dir', type=str,default='./STAGE1_PROMPTS_WEIGHT')
    parser.add_argument('--dataset_name',type=str,default='Bird')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--epochs',type=int, default=140)

    main()


