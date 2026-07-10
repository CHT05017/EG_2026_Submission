import kagglehub
import os
import shutil
import kagglehub
import kagglehub

# Download latest version
import kagglehub

# Download latest version
path = kagglehub.dataset_download("badasstechie/celebahq-resized-256x256")

print("Path to dataset files:", path)
# 然后手动复制到目标路径
target_path = "/data/dataset/chenghaotong/CelebA_HQ"
os.makedirs(target_path, exist_ok=True)

# 复制文件
if os.path.exists(path):
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(target_path, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print(f"✓ 数据集已复制到: {target_path}")