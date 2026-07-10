import json
import os
from collections import defaultdict

def create_text_from_coco(coco_annotation_file, text_output_dir):
    """
    从 COCO annotation JSON 文件创建 text 目录结构
    
    参数:
        coco_annotation_file: COCO annotation JSON 文件路径
        text_output_dir: 输出 text 目录路径
    """
    
    # 创建输出目录
    os.makedirs(text_output_dir, exist_ok=True)
    
    # 加载 JSON 文件
    print(f"加载 COCO 标注文件: {coco_annotation_file}")
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # 按 image_id 分组 caption
    caption_dict = defaultdict(list)
    
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        caption_dict[image_id].append(caption)
    
    print(f"找到 {len(caption_dict)} 张图像")
    
    # 为每个 image_id 创建 txt 文件
    for image_id, captions in caption_dict.items():
        txt_path = os.path.join(text_output_dir, f"{image_id}.txt")
        
        # 将所有 caption 写入文件（每行一个）
        with open(txt_path, 'w', encoding='utf-8') as f:
            for caption in captions:
                f.write(caption + '\n')
    
    print(f"✓ 已创建 {len(caption_dict)} 个 txt 文件到: {text_output_dir}")
    
    # 统计信息
    total_captions = sum(len(capts) for capts in caption_dict.values())
    avg_captions = total_captions / len(caption_dict)
    print(f"  - 总 caption 数: {total_captions}")
    print(f"  - 平均每张图像 caption 数: {avg_captions:.2f}")

if __name__ == '__main__':
    # 修改为你的实际路径
    coco_json = "/data/dataset/chenghaotong/COCO/annotations/captions_val2017.json"  # COCO annotation JSON
    output_dir = "/data/dataset/chenghaotong/COCO/text"                     # 输出 text 目录
    
    create_text_from_coco(coco_json, output_dir)
    
    # 验证生成结果
    print("\n示例验证:")
    example_id = list(os.listdir(output_dir))[0].replace('.txt', '')
    example_path = os.path.join(output_dir, f"{example_id}.txt")
    with open(example_path, 'r', encoding='utf-8') as f:
        print(f"文件: {example_id}.txt")
        print("内容:")
        for line in f:
            print(f"  - {line.strip()}")