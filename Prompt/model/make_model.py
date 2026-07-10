import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

ROOT_PATH = Path(__file__).resolve().parents[2] # CLIPSR/

if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from Prompt.model.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts): 
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        return x

class build_transformer(nn.Module):
    def __init__(self,dataset_name):
        super(build_transformer, self).__init__()

        if dataset_name.lower() not in ['bird','birds','celeba','coco','div2k']:
            raise FileNotFoundError('The dataset of {} does not exist!!'.format(dataset_name))

        self.cos_layer = False
        self.neck = 'bnneck'
        self.neck_feat = 'after'

        self.in_planes = 768
        self.in_planes_proj = 512

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((256-16)//16 + 1)
        self.w_resolution = int((256-16)//16 + 1)
        self.vision_stride_size = 16
        clip_model = load_clip_to_cpu('ViT-B-16', self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)

        self.prompt_learner = PromptLearner(dataset_name, clip_model.dtype, clip_model.token_embedding)



    def forward(self, x = None, index=None, get_image = False, get_text = False):
        
        if get_text == True:
            prompts = self.prompt_learner(index) 
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            if not any(p.requires_grad for p in self.prompt_learner.parameters()):  
                return text_features,prompts
            else:
                return text_features

        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            #if self.model_name == 'RN50':
            if False:
                return image_features_proj[0]
            #elif self.model_name == 'ViT-B-16':
            elif True:
                img_proj = image_features_proj[:,0]
                return image_features_proj[:,0]

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_clipsr(dataset_name):
    model = build_transformer(dataset_name)
    return model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model





class PromptLearner(nn.Module): 
    def __init__(self,dataset_name,dtype,token_embedding):
        super().__init__()
        '''
        PromptLearner for diverse datasets
        Birds, CelebA, COCO2017 supported for current use
        ''' 

        if dataset_name.lower()  in ['bird','birds','cub']:
            ctx_init='A photo of a X X X X bird.'
            num_kinds=200 # a prior
        elif dataset_name.lower() in ['celeba']:
            ctx_init='A photo of a X X X X person.'
            num_kinds=30000 # a prior
        elif dataset_name.lower() in ['coco']:
            ctx_init='A photo of a X X X X item.'
            num_kinds=123287
        elif dataset_name.lower() in ['div2k']:
            ctx_init='A photo of a X X X X item.'
            num_kinds=901
        else:
            raise NotImplementedError

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4
        
        tokenized_prompts = clip.tokenize(ctx_init).cuda() 
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) 
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_kinds, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors) 

        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])  
        self.num_class = num_kinds
        self.n_cls_ctx = n_cls_ctx

    def forward(self, indexes):# indexed by idx
        indexes=torch.tensor(indexes)
        cls_ctx = self.cls_ctx[indexes] # (B, 4,512)

        b = indexes.shape[0]
        
        prefix = self.token_prefix.expand(b, -1, -1) # (1,5,512)
        #cls_ctx=cls_ctx.unsqueeze(0).expand(1,-1,-1) # (1,4,512)
        
    
        suffix = self.token_suffix.expand(b, -1, -1) 
            
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        ) 

        #print(prompts.shape) # (B,77,512)
        

        return prompts 


