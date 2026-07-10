import os
import os.path as osp
import pickle

import glob
class COCO_dataset(object):
    root='/data/dataset/chenghaotong/COCO'
    def __init__(self):
        super().__init__()

        self.train_img_dir=osp.join(self.root, 'train')
        self.test_img_dir=osp.join(self.root, 'test')

        if not osp.exists(self.train_img_dir):
            raise FileNotFoundError('The path ur looking for seems missing...')        

        self.train_set,self.fake_mapping=self.process()


    def process_names(self,item):
        return item+'.jpg'


    def process(self):
        train_imgs_1=sorted(glob.glob(osp.join(self.train_img_dir,'*.jpg')))
        train_imgs_2=sorted(glob.glob(osp.join(self.test_img_dir,'*.jpg')))

        train_imgs=train_imgs_2+train_imgs_1
        #cat_names=[(((item.split('/'))[0]).split('.'))[-1] for item in eff_names]
        #cat_names=set(cat_names) # len=150        

        train_imgs=sorted(train_imgs)

        names=[((osp.basename(p)).split('.'))[0] for p in train_imgs]
        names=sorted(names,key=int)
        
        fake_mapping = {img_name: label for label, img_name in enumerate(names)}
        #fake_mapping={(osp.basename(t_img).split('.'))[-2]: \
        #              int((osp.basename(t_img).split('.'))[-2]) for t_img in train_imgs}

        #print(fake_mapping)
        ##print(len(fake_mapping.keys())) # 123287
        #exit()

        return train_imgs,fake_mapping




if __name__ == '__main__':
    bird=COCO_dataset()
    #print(bird.train_set)

# python Prompt/dataset/COCO.py
