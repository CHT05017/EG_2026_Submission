
import os.path as osp
import glob
import shutil
import os
from tqdm import tqdm

root='/data/ckpt/chenghaotong/CVIDL2026/CLIPSR/EVAL_imgs/bird/test/TEST_bird_256_2025_12_27_11_44_26/x16_CUB_OURS_250'

img_paths=sorted(glob.glob(osp.join(root,'*.jpg')))


save_dir='/data/ckpt/chenghaotong/x16_CUB_LR_2'
if not osp.exists(save_dir):
    os.makedirs(save_dir,exist_ok=True)

for p in tqdm(img_paths):
    name=osp.basename(p)
    name2=name.split('.')[-2]
    if name2[-2:]=='LR':
        new_path=osp.join(save_dir,name)
        shutil.copy(p,new_path)

# python a.py


    

