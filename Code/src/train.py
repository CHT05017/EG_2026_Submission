import os, sys
import os.path as osp
import time
import random
import argparse
import numpy as np
from PIL import Image
import pprint

import warnings
warnings.filterwarnings(action='ignore',category=UserWarning)

from pathlib import Path

os.environ['OMP_NUM_THREADS']='5'
os.environ['MKL_NUM_THREADS']='5'


ROOT_PATH = Path(__file__).resolve().parents[2] # CLIPSR/

if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image,make_grid
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data.distributed import DistributedSampler
import multiprocessing as mp

#ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
#sys.path.insert(0, ROOT_PATH)

#sys.path.append('/opt/data/private/carr/code/lib')
from Code.lib.utils import mkdir_p,get_rank,merge_args_yaml,get_time_stamp,save_args
from Code.lib.utils import load_models_opt,save_models_opt,save_models,load_npz,params_count
from Code.lib.perpare import prepare_dataloaders,prepare_models
from Code.lib.modules import sample_one_batch as sample, test as test, train as train
from Code.lib.datasets import get_fix_data

from Prompt.load import Text_features_extractor
def extractor_define(args):
    extractor=Text_features_extractor(dataset_name=((((args.cfg_file).split('/'))[-1]).split('.'))[0])
    return extractor


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='TextAuxiliaryImgSR')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='/home/chenghaotong/CLIPSR_ver3/Code/cfg/DIV2K.yml',
                        help='optional config file')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of workers(default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--stamp', type=str, default='normal',
                        help='the stamp of model')
    parser.add_argument('--pretrained_model_path', type=str, default='model',
                        help='the model for training')
    parser.add_argument('--log_dir', type=str, default='./LOGS', 
                        help='file path to log directory')
    parser.add_argument('--model', type=str, default='net',
                        help='the model for training')
    parser.add_argument('--state_epoch', type=int, default=1,
                        help='state epoch')
    parser.add_argument('--batch_size', type=int, default=16 ,
                        help='batch size')
    parser.add_argument('--train', type=str, default='True',
                        help='if train model')
    parser.add_argument('--mixed_precision', type=str, default='False',
                        help='if use multi-gpu')
    parser.add_argument('--multi_gpus', type=str, default='False',
                        help='if use multi-gpu')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='gpu id')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--scaler', default=4, type=int,
                        help='2,3,4,8')
    parser.add_argument('--random_sample', action='store_true',default=True, 
                        help='whether to sample the dataset with random sampler')
    args = parser.parse_args()
    return args


def main(args):

    #--------------------------------------model,log,image save path------------------------------------
    time_stamp = get_time_stamp()
    stamp = '_'.join([str(args.model),'nf'+str(args.nf),str(args.stamp),str(args.CONFIG_NAME),str(args.imsize),time_stamp])
    args.model_save_file = osp.join(ROOT_PATH, 'saved_models', str(args.CONFIG_NAME), stamp,f'x{args.scaler}')
    log_dir = args.log_dir
    if log_dir == 'new':

        log_dir = osp.join(ROOT_PATH, 'logs/{0}'.format(osp.join(str(args.CONFIG_NAME), 'train', stamp)))

        mkdir_p(log_dir)

    args.img_save_dir = osp.join(ROOT_PATH, 'imgs/{0}'.format(osp.join(str(args.CONFIG_NAME), 'train', stamp,f'x{args.scaler}')))
    #------------------------------------------------------------------------------------------------------
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        mkdir_p(osp.join(ROOT_PATH, 'logs'))
        mkdir_p(args.model_save_file)
        mkdir_p(args.img_save_dir)

    # prepare TensorBoard
    if (args.multi_gpus==True) and (get_rank() != 0):
        writer = None
    else:
        writer = SummaryWriter(log_dir)
 

    #============================================ prepare dataloader, models, data
    
    train_dl, valid_dl ,train_ds, valid_ds, sampler = prepare_dataloaders(args)
    
    extractor=extractor_define(args)
    print("Stage1 prompt ready!!")


    CLIP4trn, CLIP4evl, image_encoder, text_encoder, netG, netD, netC = prepare_models(args)

    print('**************G_paras: ',params_count(netG))
    print('**************D_paras: ',params_count(netD)+params_count(netC))
    print('**************else: ', params_count(CLIP4trn) + params_count(CLIP4evl)+ params_count(image_encoder)+ params_count(text_encoder))
    GT,LR, fixed_sent, fixed_words,fixed_z = get_fix_data(train_dl, valid_dl,text_encoder, extractor,args)


    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        fixed_grid = make_grid(GT.cpu(), nrow=8, normalize=True)
        #writer.add_image('fixed images', fixed_grid, 0)
        img_name = 'gt.png'
        img_save_path = osp.join(args.img_save_dir, img_name)
        LR_img_name="LR.png"
        LR_img_save_path = osp.join(args.img_save_dir, LR_img_name)
        vutils.save_image(GT.data, img_save_path, nrow=8, normalize=True)
        vutils.save_image(LR, LR_img_save_path, nrow=8, normalize=True)
        print("----finish---")

    # ############################prepare optimizer,set lr=================================
    D_params = list(netD.parameters()) + list(netC.parameters())
    optimizerD = torch.optim.Adam(D_params, lr=args.lr_d, betas=(0.0, 0.9))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g, betas=(0.0, 0.9))
    if args.mixed_precision==True:
        scaler_D = torch.cuda.amp.GradScaler(growth_interval=args.growth_interval)
        scaler_G = torch.cuda.amp.GradScaler(growth_interval=args.growth_interval)
    else:
        scaler_D = None
        scaler_G = None

    # m1, s1 = load_npz(args.npz_path)
 
    start_epoch = 1
    # ==================================================load from checkpoint===================================
  
    if args.state_epoch!=1:
        start_epoch = args.state_epoch + 1
        path = osp.join(args.pretrained_model_path, 'state_epoch_%03d.pth'%(args.state_epoch))
        netG, netD, netC, optimizerG, optimizerD = load_models_opt(netG, netD, netC, optimizerG, optimizerD, path, args.multi_gpus)
       
    # print args
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        pprint.pprint(args)
        arg_save_path = osp.join(log_dir, 'args.yaml')
        save_args(arg_save_path, args)


        print("Start Training")
    # ===================================================================Start training
    test_interval,gen_interval,save_interval = args.test_interval,args.gen_interval,args.save_interval
 
    for epoch in range(start_epoch, args.max_epoch, 1):
        #if (args.multi_gpus==True):
        #    sampler.set_epoch(epoch)
        start_t = time.time()

        # training
        args.current_epoch = epoch
        torch.cuda.empty_cache()

       
        train(train_dl, netG, netD, netC, text_encoder,extractor,image_encoder, optimizerG, optimizerD, scaler_G, scaler_D, writer,args)
        torch.cuda.empty_cache()
        # ==============================================save============================================================
        if epoch%save_interval==0:
            save_models_opt(netG, netD, netC, optimizerG, optimizerD, epoch, args.multi_gpus, args.model_save_file)
            torch.cuda.empty_cache()
        # =============================================sample===============================================
        if epoch%gen_interval==0:
        
            sample(LR, fixed_sent, netG, args.multi_gpus, epoch, args.img_save_dir)
            torch.cuda.empty_cache()
        
        # ============================================test===================================================
        if epoch%test_interval==0:

            PSNR = test(args,valid_dl, text_encoder, extractor,netG,None, CLIP4evl, args.device, epoch, args.max_epoch, args.sample_times, args.z_dim, args.batch_size)
            torch.cuda.empty_cache()
            print("---------------------",PSNR,"-----------------------------")


        if (args.multi_gpus==True) and (get_rank() != 0):
            None
        else:
            if epoch%test_interval==0:
                writer.add_scalar('PSNR', PSNR, epoch)
                # writer.add_scalar('CLIP_Score', TI_score, epoch)
                print('The %d epoch PSNR: %.2f' % (epoch,PSNR))
            end_t = time.time()
            print('The epoch %d costs %.2fs'%(epoch, end_t-start_t))
            print("*"*40)


if __name__ == "__main__":
    args = merge_args_yaml(parse_args())
    # set seed
    if args.manual_seed is None:
        args.manual_seed = 100
        #args.manualSeed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.manual_seed)
        if args.multi_gpus:
            # DataParallel 模式：使用所有可用 GPU
            args.device = torch.device("cuda")
            args.local_rank = -1
        else:
            # 单 GPU 模式
            torch.cuda.set_device(args.gpu_id)
            args.device = torch.device("cuda")
            args.local_rank = -1
    else:
        args.device = torch.device('cpu')
    main(args)

