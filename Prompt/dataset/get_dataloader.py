import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image
import sys
from pathlib import Path
ROOT_PATH = Path(__file__).resolve().parents[2] # CLIPSR/

if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))


from Prompt.dataset.Bird import  Bird_dataset
from Prompt.dataset.CelebA import CelebA_dataset
from Prompt.dataset.COCO import COCO_dataset
from Prompt.dataset.DIV2K import DIV2K_dataset

_factory={
    'Bird':Bird_dataset(),
    'Birds':Bird_dataset(),
    'COCO':COCO_dataset(),
    'CelebA':CelebA_dataset(),
    'DIV2K':DIV2K_dataset()
}


def create(name):
    got=_factory[name]
    return got


class Preprocessor(Dataset):
    def __init__(self, dataset, transform=None): # dataset应该是一个具体的东西，例如是market数据集的train_set或者lpw数据集的test_set
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        # print(self.dataset)

        fpaths=self.dataset[index]
        img = Image.open(fpaths).convert('RGB')

        img_transformed = self.transform(img)


        return img_transformed, fpaths

import torchvision.transforms as T
from Prompt.dataset.sampler import RandomMultipleGallerySampler

def get_data(args,name):
    got_dst=create(name)
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    transform = T.Compose([
            T.Resize((256,256)),
            T.ToTensor(),
            normalizer
            
        ])
    
    train_set=got_dst.train_set
    
    sampler=RandomMultipleGallerySampler(train_set) if name.lower() in ['bird','birds'] else None

    stage1_loader=DataLoader(
        Preprocessor(train_set,transform),
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=False
    )
    #if not name.lower() in ['bird','birds']:
    #    fake_mapping=None
    #else:
    fake_mapping=got_dst.fake_mapping

    return stage1_loader,fake_mapping
    
    
if __name__ == '__main__':
    stage1_loader = get_data('CelebA')
    
    for batch_idx, (img_tensor, fpaths) in enumerate(stage1_loader):
        print(f"\n=== Batch {batch_idx} ===")
        print(f"Image tensor shape: {img_tensor.shape}")
        print(f"Image tensor dtype: {img_tensor.dtype}")
        print(f"Image tensor range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
        print(f"Filepaths count: {len(fpaths)}")
        print(f"First 3 filepaths: {fpaths[:3]}")
        
        if batch_idx >= 2:  
            break
# python Prompt/dataset/get_dataloader.py

    


    





