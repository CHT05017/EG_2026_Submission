import warnings
warnings.filterwarnings(action='ignore',category=FutureWarning)

import os.path as osp
import sys

import torch
from pathlib import Path
ROOT_PATH = Path(__file__).resolve().parents[2] # CLIPSR/
import pickle
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

try:
    from Prompt.model.make_model import make_clipsr
except ModuleNotFoundError:
    from model.make_model import make_clipsr

def _set_stage2_mode(model):
    '''
    GOAL:
    - to freeze the clipreid prompt and text encoder, let query and image encoder trainable
    '''
    # to get base models
    base = model

    # clipreid prompt & text encoder frozen
    if hasattr(base, 'prompt_learner'):
        for p in base.prompt_learner.parameters():
            p.requires_grad = False
    if hasattr(base, 'text_encoder'):
        for p in base.text_encoder.parameters():
            p.requires_grad = False

    for m in [getattr(base, 'image_encoder', None), getattr(base, 'classifier', None),getattr(base, 'classifier_proj', None),getattr(base, 'bottleneck', None),getattr(base, 'bottleneck_proj', None)]:
        if m is not None:
            for p in m.parameters():
                p.requires_grad = False
import glob
class Text_features_extractor():
    def __init__(self,dataset_name):
        super().__init__()
        self.dataset_name=dataset_name
        
        self.prompt_root='/home/chenghaotong/CLIPSR_ver3/Prompt/STAGE1_PROMPTS_WEIGHT/'
        
        if self.dataset_name.lower() in ['bird','birds']:
            self.prompt_path = osp.join(self.prompt_root,'Bird_prompt_350.pth')
            self.img_dir='/data/dataset/chenghaotong/Birds/CUB_200_2011/images'
            self.fake_mapping=self.process()
        elif self.dataset_name.lower() in ['celeba']:
            self.root='/data/dataset/chenghaotong/CelebA/'
            self.img_dir=osp.join(self.root,'images')
            self.prompt_path=osp.join(self.prompt_root,'CelebA_prompt_60.pth')
            self.fake_mapping=self.process()
        elif self.dataset_name.lower() in ['coco']:
            self.prompt_path=osp.join(self.prompt_root,'COCO_prompt_50.pth')
            self.fake_mapping=self.process()
        elif self.dataset_name.lower() in ['div2k']:
            self.prompt_path=osp.join(self.prompt_root,'DIV2K_prompt_300.pth')
            self.fake_mapping=self.process()


        self.prompt=self.load_prompt(self.prompt_path)

        self.model=make_clipsr(self.dataset_name)
        self.model.to('cuda')


        self.prompt_initialize(self.model,self.prompt)
        print('--> model prompt successfully initialized using {}...'.format(self.prompt_path))
        _set_stage2_mode(self.model)

        
    def process_names(self,item):

        if item.startswith('aV'):
            item=item[2:]
        elif item.startswith('V'):
            item=item[1:]

        return item+'.jpg'

    def process_names4cele(self,item):
        return item+'.jpg'

    def process(self):

        if self.dataset_name.lower() in ['bird','birds']:
            with open('/data/dataset/chenghaotong/Birds/train/filenames.pickle','r') as f:
                names=f.readlines()

            with open('/data/dataset/chenghaotong/Birds/test/filenames.pickle','r') as f_t:
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
            
        if self.dataset_name.lower() in ['celeba']:
            with open(osp.join(self.root,'train','filenames.pickle'),'rb') as f:
                names=pickle.load(f)

            with open(osp.join(self.root,'test','filenames.pickle'),'rb') as f_t:
                test_names=pickle.load(f_t)

            train1_names=[self.process_names4cele(item) for item in names]
            train2_names=[self.process_names4cele(item) for item in test_names]

            train_imgs_1=[osp.join(self.img_dir,item) for item in train1_names]
            train_imgs_2=[osp.join(self.img_dir,item) for item in train2_names]

            train_imgs=train_imgs_2+train_imgs_1
            #cat_names=[(((item.split('/'))[0]).split('.'))[-1] for item in eff_names]
            #cat_names=set(cat_names) # len=150        

            train_imgs=sorted(train_imgs)
            
            #fake_mapping = {dirname: label for label, dirname in enumerate(unique_names)}
            fake_mapping={(osp.basename(t_img).split('.'))[-2]: \
                        int((osp.basename(t_img).split('.'))[-2]) for t_img in train_imgs}

            
        if self.dataset_name.lower() in ['coco','coco2017']:
            train_imgs_1=sorted(glob.glob(osp.join('/data/dataset/chenghaotong/COCO/train','*.jpg')))
            train_imgs_2=sorted(glob.glob(osp.join('/data/dataset/chenghaotong/COCO/test','*.jpg')))

            train_imgs=train_imgs_2+train_imgs_1
            #cat_names=[(((item.split('/'))[0]).split('.'))[-1] for item in eff_names]
            #cat_names=set(cat_names) # len=150        

            train_imgs=sorted(train_imgs)

            names=[((osp.basename(p)).split('.'))[0] for p in train_imgs]
            names=sorted(names,key=int)
            
            fake_mapping = {img_name: label for label, img_name in enumerate(names)}
            ##print(fake_mapping)
            #exit()

        if self.dataset_name.lower() in ['div2k']:
            train_imgs_1=sorted(glob.glob(osp.join('/data/dataset/chenghaotong/DIV2K/train','*.jpeg')))
            train_imgs_2=sorted(glob.glob(osp.join('/data/dataset/chenghaotong/DIV2K/test','*.jpeg')))

            train_imgs=train_imgs_2+train_imgs_1
            #cat_names=[(((item.split('/'))[0]).split('.'))[-1] for item in eff_names]
            #cat_names=set(cat_names) # len=150        

            train_imgs=sorted(train_imgs)

            #names=[((osp.basename(p)).split('.'))[0] for p in train_imgs]
            #names=sorted(names,key=int)
            names=[(osp.basename(p).split('.'))[0] for p in train_imgs]
            
            fake_mapping = {name: int(name) for name in names}

        return fake_mapping

    def load_prompt(self,path):
        state = torch.load(path, map_location='cpu')

        if isinstance(state, dict) and 'state_dict' in state:
            return state['state_dict']
        return state

    def prompt_initialize(self,base, prompt_state):
        prompt_learner = base.prompt_learner
        prompt_learner.load_state_dict(prompt_state, strict=False)
        prompt_learner.num_class = prompt_state['cls_ctx'].shape[0]

    def get_text_features(self,keys): 
        '''
        Docstring for get_text_features
        
        :param key: {048.European_Goldfinch/European_Goldfinch_0067_794637}, it's a list
        '''

        #print(keys)
        #exit()
        pseudo_keys=[(key.split('/'))[0] for key in keys]
        pseudo_labels=[self.fake_mapping[pseudo_key] for pseudo_key in pseudo_keys]
        #print(pseudo_labels) [31,31,31,31,31]
        

        text_feat,prompts=self.model(index=pseudo_labels,get_text=True)

        #print(text_feat.shape) # (B,512)
        #print(prompts.shape) # (B,77,512)
        #exit()

        sent_emb=text_feat
        words_emb=prompts

        return sent_emb,words_emb


if __name__== '__main__':
    extractor = Text_features_extractor(dataset_name='Bird')
    keys=["048.European_Goldfinch/a","048.European_Goldfinch/a","048.European_Goldfinch/a","048.European_Goldfinch/a","048.European_Goldfinch/a"]
    text_feat=extractor.get_text_features(keys)



# python Prompt/load.py

