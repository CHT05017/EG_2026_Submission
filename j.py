
import os
import os.path as osp
import glob

root='/data/dataset/chenghaotong/DIV2K'
img_ps=sorted(glob.glob(osp.join(root,'train','*.jpeg')))

txt_dir=osp.join(root,'text')
if not osp.exists(txt_dir):
    os.makedirs(txt_dir,exist_ok=True)

from tqdm import tqdm

for p in tqdm(img_ps):
    name=osp.basename(p).split('.')[-2]
    txt_name=name+'.txt'
    txt_p=osp.join(txt_dir,txt_name)

    #if not osp.exists(txt_p):
    #    os.makedirs(txt_p,exist_ok=True)

    with open(txt_p,'w') as f:
        f.write("A high quality photo of an item, clear details.")

# python j.py
    


