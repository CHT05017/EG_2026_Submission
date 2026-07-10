import os
import sys
import torch
import torchvision.models as models

# 步骤 1：设置缓存目录
os.environ['TORCH_HOME'] = '/home/chenghaotong/.cache/torch'

# 步骤 2：设置国内镜像（在 import torch 之后）
# 修改 hub 配置以使用国内镜像
def set_mirror():
    """设置国内镜像"""
    # 方法 1：修改 hub 的默认 URL
    import torch.hub as hub
    
    # 清华镜像
    #hub.MASTER_BRANCH = 'master'
    #hub.BRANCH = 'master'
    
    # 设置阿里云镜像（如果支持）
    os.environ['TORCH_MODEL_ZOO'] = 'https://mirrors.aliyun.com/pytorch-wheels/pytorch_model_zoo/'

set_mirror()

# 现在下载
print("开始下载 VGG16...")
vgg16 = models.vgg16(pretrained=True)
print("✓ VGG16 下载完成")