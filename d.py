import os
import glob
import os.path as osp

img_path='/data/dataset/chenghaotong/COCO/train'
img_path_2='/data/dataset/chenghaotong/COCO/test'

from tqdm import tqdm
import shutil

img_paths = sorted(glob.glob(osp.join(img_path,'*.jpg')))

img_paths2 = sorted(glob.glob(osp.join(img_path_2,'*.jpg')))

new_path='/data/dataset/chenghaotong/COCO/garbage'
if not osp.exists(new_path):
    os.makedirs(new_path,exist_ok=True)
for idx,p in tqdm(enumerate(img_paths),total=len(img_paths)):
    old=(osp.basename(p).split('.'))[0]
    if old[:3]=='000':
    #old=str(int(old))
    #new_name=f'{old}.jpg'
    #new_path=osp.join(img_path,new_name)
        new=osp.join(new_path,osp.basename(p))
        os.remove(p)
    else:
        continue


for idx,p in tqdm(enumerate(img_paths2),total=len(img_paths2)):
    old=(osp.basename(p).split('.'))[0]
    
    old=str(int(old))
    new_name=f'{old}.jpg'
    new_path=osp.join(img_path_2,new_name)
    shutil.copy(p,new_path)

for idx,p in tqdm(enumerate(img_paths2),total=len(img_paths2)):
    old=(osp.basename(p).split('.'))[0]
    if old[:3]=='000':

    #old=str(int(old))
    #new_name=f'{old}.jpg'
    #new_path=osp.join(img_path_2,new_name)
        os.remove(p)
    else:
        continue


