import os
import sys
import time
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import warnings
warnings.filterwarnings(action='ignore',category=FutureWarning)
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
#import clip as clip

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2] # CLIPSR/

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import CLIP.clip.clip as clip

import torch.nn.functional as F
from Code.lib.utils_image import imresize

def get_fix_data(train_dl, test_dl, text_encoder, extractor,args):
    fixed_image_train,LR1, _, _, fixed_sent_train, fixed_word_train, fixed_key_train = get_one_batch_data(train_dl, text_encoder, extractor,args)
    fixed_image_test,LR2, _, _, fixed_sent_test, fixed_word_test, fixed_key_test= get_one_batch_data(test_dl, text_encoder, extractor,args)
    fixed_image = torch.cat((fixed_image_train, fixed_image_test), dim=0)
    fixed_sent = torch.cat((fixed_sent_train, fixed_sent_test), dim=0)
    LR = torch.cat((LR1, LR2), dim=0)
    fixed_word = torch.cat((fixed_word_train, fixed_word_test), dim=0)
    fixed_noise = torch.randn(fixed_image.size(0), args.z_dim).to(args.device)


    return fixed_image,LR,fixed_sent, fixed_word, fixed_noise


def get_one_batch_data(dataloader, text_encoder, extractor,args):
    data = next(iter(dataloader))
    imgs,LR, captions, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(args,data, text_encoder, extractor,args.device)
    return imgs,LR, captions, CLIP_tokens, sent_emb, words_embs, keys



def prepare_data(args,data, text_encoder, extractor,device):
    imgs,LR, captions, CLIP_tokens, keys = data # 在这里接入
    imgs, CLIP_tokens = imgs.to(device), CLIP_tokens.to(device)
    #sent_emb, words_embs = encode_tokens(text_encoder, CLIP_tokens)

    extractor=extractor
    
    #print(sent_emb.shape) # (64,512) -> 整个句子语义表示
    #print(words_embs.shape) # (64,77,512) -> 逐个词语的语义表示
    #print(words_embs)

    LR=LR.to(device)
    #print(LR.shape) # (64,3,64,64)

    sent_emb,words_embs=extractor.get_text_features(list(keys))

    #print(sent_emb.shape)
    #print(words_emb.shape)
    
    return imgs, LR,captions, CLIP_tokens, sent_emb, words_embs, keys


def encode_tokens(text_encoder, caption):
    # encode text
    with torch.no_grad():
        sent_emb,words_embs = text_encoder(caption)
        sent_emb,words_embs = sent_emb.detach(), words_embs.detach()
    return sent_emb, words_embs 


def get_imgs(img_path, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
    if transform is not None:
        img = transform(img)
    if normalize is not None:
        img = normalize(img)
    
    return img


def get_caption(cap_path,clip_info): # 如果有多行文本，他是随机选择一个文本
    eff_captions = []
    with open(cap_path, "r") as f:
        captions = f.read().encode('utf-8').decode('utf8').split('\n')
    for cap in captions:
        if len(cap) != 0:
            eff_captions.append(cap)
    sent_ix = random.randint(0, len(eff_captions))
    caption = eff_captions[sent_ix]
    
    tokens = clip.tokenize(caption,truncate=True) # [1,77]

    #print((tokens[0]).shape) # [77]
    
    return caption, tokens[0]


################################################################
#                    Dataset
################################################################
class TextImgDataset(data.Dataset):
    def __init__(self, split, transform=None, args=None):
        self.transform = transform
        self.clip4text = args.clip4text
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset_name
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.split=split
        
        if self.data_dir.find('Birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        self.split_dir = os.path.join(self.data_dir, split)
        self.filenames = self.load_filenames(self.data_dir, split)
        self.number_example = len(self.filenames)

        self.sr_scale=int(args.scaler)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        return filename_bbox



    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)

        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def __getitem__(self, index):

        key = self.filenames[index] 
        #print(index)
        

        ####

        key = key.removesuffix('.jpg')
        data_dir = self.data_dir
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
        else:
            bbox = None

        if self.dataset_name.lower().find('coco') != -1:
            if self.split=='train':
                img_name = '%s/train/%s.jpg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key)
            else:
                img_name = '%s/test/%s.jpg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key)

        elif self.dataset_name.lower().find('cele') != -1:
            if self.split=='train':
                img_name = '%s/image/%s.jpg' % (data_dir, key)
                text_name = '%s/text/text/%s.txt' % (data_dir, key)

            else:
                img_name = '%s/image/%s.jpg' % (data_dir, key)
                text_name = '%s/text/text/%s.txt' % (data_dir, key)
        elif self.dataset_name.lower().find('div2k') != -1:
            if self.split=='train':
                img_name = '%s/train/%s.jpeg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key)
            else:
                img_name = '%s/test/%s.jpeg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key)
        else:
            img_name = '%s/CUB_200_2011/images/%s.jpg' % (data_dir, key)
            text_name = '%s/text/%s.txt' % (data_dir, key)

        imgs = get_imgs(img_name, bbox, self.transform, normalize=self.norm)
        LR =imresize(imgs, 1/self.sr_scale, True)
        #print(text_name) # /data/dataset/chengahotong/.../048.glodfish.txt

        caps,tokens = get_caption(text_name,self.clip4text)
        
        #print(caps)
        #print(tokens.shape)
        return imgs,LR, caps, tokens, key

    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':
    import torch
    import torchvision.transforms as transforms
    from argparse import Namespace
    
    args = Namespace(
        data_dir='/data/dataset/chenghaotong/COCO',
        dataset_name='coco',
        clip4text={'src': 'clip', 'type': 'ViT-B/32'},
        device='cpu'
    )
    
    # 创建变换
    transform = transforms.Compose([
        transforms.Resize(276),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
    ])
    
    dataset = TextImgDataset(split='train', transform=transform, args=args)
    print(f"{len(dataset)}")
    
    imgs, lr, caps, tokens, key = dataset[0]
    print(f"{caps}")    