import os
import pickle
from pathlib import Path

def generate_filenames_pickle(image_dir, output_dir):
    """
    从图像目录生成 filenames.pickle
    
    Args:
        image_dir: 图像目录路径，例如 /data/train/
        output_dir: 输出 pickle 文件的目录，例如 /data/train/
    """
    
    # 获取所有 jpg 文件
    image_files = sorted([f[:-5] for f in os.listdir(image_dir) if f.endswith('.jpeg')])
    #print(image_files[:4])
    #exit()
    
    print(f"找到 {len(image_files)} 张图像")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存 pickle 文件
    pickle_path = os.path.join(output_dir, 'filenames.pickle')
    with open(pickle_path, 'wb') as f:
        pickle.dump(image_files, f)
    
    print(f"✓ 生成成功: {pickle_path}")
    print(f"  示例: {image_files[:3]}")

# 使用
generate_filenames_pickle('/data/dataset/chenghaotong/DIV2K/test/', '/data/dataset/chenghaotong/DIV2K/test/')

