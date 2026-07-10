import os
import glob
import os.path as osp

img_path='/data/dataset/chenghaotong/COCO/test'

from tqdm import tqdm
import shutil

img_paths = sorted(glob.glob(osp.join(img_path,'*.jpg')))

new_p='/data/dataset/chenghaotong/CelebA/image'
cnt=0
for idx,p in tqdm(enumerate(img_paths)):
    #old=osp.basename(p)
    #new_name=f'{idx}.jpg'
    #new_path=osp.join(new_p,new_name)
    #shutil.copy(p,new_path)
    cnt += 1
print(cnt)