import os
import copy
import re
import glob
import os.path as osp
import warnings
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[2] # CLIPSR/

if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))


class Bird_dataset(object):
    root='/data/dataset/chenghaotong/Birds'
    def __init__(self):
        super().__init__()

        self.img_dir=osp.join(self.root, 'CUB_200_2011','images')
        self.train_pkl=osp.join(self.root,'train','filenames.pickle')
        self.test_pkl=osp.join(self.root, 'test','filenames.pickle')
        self.prompt_train_imgs=osp.join(self.root,'prompt_train_set')

        if not osp.exists(self.prompt_train_imgs):
            os.makedirs(self.prompt_train_imgs, exist_ok=True)

        if not osp.exists(self.img_dir):
            raise FileNotFoundError('The path ur looking for seems missing...')        

        self.train_set,self.fake_mapping=self.process()

        

    def process_names(self,item):

        if item.startswith('aV'):
            item=item[2:]
        elif item.startswith('V'):
            item=item[1:]

        return item+'.jpg'


    def process(self):

        with open(self.train_pkl,'r') as f:
            names=f.readlines()

        with open(self.test_pkl,'r') as f_t:
            test_names=f_t.readlines()

        eff_names=[item.rstrip('\n') for item in names if item.startswith(('aV','V'))]
        eff_names=[self.process_names(item) for item in eff_names if item.startswith(('V','aV'))]

        eff_names_test=[item.rstrip('\n') for item in test_names if item.startswith(('aV','V'))]
        eff_names_test=[self.process_names(item) for item in eff_names_test if item.startswith(('V','aV'))]

        train_imgs_1=[osp.join(self.img_dir,item) for item in eff_names]
        train_imgs_2=[osp.join(self.img_dir,item) for item in eff_names_test]

        train_imgs=train_imgs_2+train_imgs_1
        #cat_names=[(((item.split('/'))[0]).split('.'))[-1] for item in eff_names]
        #cat_names=set(cat_names) # len=150        

        train_imgs=sorted(train_imgs)
        unique_dirs =sorted({(item.split('/'))[-2] for item in train_imgs})  

        fake_mapping = {dirname: label for label, dirname in enumerate(unique_dirs)}

        return train_imgs,fake_mapping




if __name__ == '__main__':
    bird=Bird_dataset()
    #print(bird.train_set)

# python Prompt/dataset/Bird.py



