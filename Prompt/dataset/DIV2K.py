import os
import os.path as osp
import pickle


class DIV2K_dataset(object):
    root='/data/dataset/chenghaotong/DIV2K'
    def __init__(self):
        super().__init__()

        #self.img_dir=osp.join(self.root, 'image')
        self.train_img_dir=osp.join(self.root,'train')
        self.test_img_dir=osp.join(self.root,'test')
        
        self.train_pkl=osp.join(self.root,'train','filenames.pickle')
        self.test_pkl=osp.join(self.root, 'test','filenames.pickle')
        self.prompt_train_imgs=osp.join(self.root,'prompt_train_set')

        if not osp.exists(self.prompt_train_imgs):
            os.makedirs(self.prompt_train_imgs, exist_ok=True)

        #if not osp.exists(self.img_dir):
        #    raise FileNotFoundError('The path ur looking for seems missing...')        

        self.train_set,self.fake_mapping=self.process()


    def process_names(self,item):
        return item+'.jpeg'


    def process(self):

        with open(self.train_pkl,'rb') as f:
            names=pickle.load(f)

        with open(self.test_pkl,'rb') as f_t:
            test_names=pickle.load(f_t)

        #print(test_names)
        #exit()

        train1_names=[self.process_names(item) for item in names]
        train2_names=[self.process_names(item) for item in test_names]


        train_imgs_1=[osp.join(self.train_img_dir,item) for item in train1_names]
        train_imgs_2=[osp.join(self.test_img_dir,item) for item in train2_names]

        train_imgs=train_imgs_2+train_imgs_1
        #cat_names=[(((item.split('/'))[0]).split('.'))[-1] for item in eff_names]
        #cat_names=set(cat_names) # len=150        
        train_imgs=sorted(train_imgs)
        
        #fake_mapping = {dirname: label for label, dirname in enumerate(unique_names)}
        fake_mapping={(osp.basename(t_img).split('.'))[-2]: \
                      int((osp.basename(t_img).split('.'))[-2]) for t_img in train_imgs}
        
        return train_imgs,fake_mapping




if __name__ == '__main__':
    bird=DIV2K_dataset()
    #print(bird.train_set)

# python Prompt/dataset/DIV2K.py
