import os
import pickle
import random
from pathlib import Path

def generate_sampled_pickle(image_dir, output_dir, num_samples=20000, seed=42):
    """
    从大数据集中随机采样生成 filenames.pickle
    
    Args:
        image_dir: 原始图像目录路径，例如 /data/COCO/train/
        output_dir: 输出 pickle 文件的目录，例如 /data/COCO/train/
        num_samples: 采样的图像数量，默认 20000
        seed: 随机种子，保证可复现性
    """
    
    # 设置随机种子，保证结果可复现
    random.seed(seed)
    
    # 获取所有 jpg 文件
    all_images = sorted([f[:-4] for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    print(f"找到总共 {len(all_images)} 张图像")
    
    # 检查采样数量是否合法
    if num_samples > len(all_images):
        print(f"⚠️  警告: 采样数量({num_samples})大于总数({len(all_images)})")
        print(f"将使用全部 {len(all_images)} 张图像")
        sampled_images = all_images
    else:
        # 随机采样
        sampled_images = random.sample(all_images, num_samples)
        print(f"✓ 随机采样了 {len(sampled_images)} 张图像")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存 pickle 文件
    pickle_path = os.path.join(output_dir, 'filenames.pickle')
    with open(pickle_path, 'wb') as f:
        pickle.dump(sampled_images, f)
    
    print(f"✓ 生成成功: {pickle_path}")
    print(f"  示例前3个: {sampled_images[:3]}")
    print(f"  示例后3个: {sampled_images[-3:]}")
    
    return sampled_images

# 使用示例
if __name__ == '__main__':
    # 从 COCO 训练集随机采样 20000 张
    sampled_images = generate_sampled_pickle(
        image_dir='/data/dataset/chenghaotong/COCO/train/',
        output_dir='/data/dataset/chenghaotong/COCO/train/',
        num_samples=20000,
        seed=42
    )