import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def vis_att(model, input_image, layer_index=11, save_path="attention_map.png", alpha=0.5):
    model.eval()
    
    try:
        # Handle DataParallel
        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
            
        # Access the transformer block
        block = model.mapping.CLIP_ViT.transformer.resblocks[layer_index]
    except AttributeError:
        print("Could not find the transformer block. Check model structure.")
        return

    attn_weights = block.attn_weights
    
    if attn_weights is None:
        print("No attention weights found. Please run a forward pass with the model first.")
        return

    # Take the first item in the batch
    attn_map = attn_weights[0]  # (SeqLen, SeqLen)
    
    # Determine grid size
    seq_len = attn_map.shape[0]
    grid_size = int(np.sqrt(seq_len - 1))
    if grid_size * grid_size != seq_len - 1:
        grid_size = int(np.sqrt(seq_len - 2))
        if grid_size * grid_size != seq_len - 2:
            print(f"Could not determine grid size from sequence length {seq_len}")
            return

    # Extract attention to patches (CLS token attention to patches)
    patch_attn = attn_map[0, 1 : 1 + grid_size*grid_size]
    
    # Reshape to grid
    patch_attn = patch_attn.reshape(grid_size, grid_size).detach().cpu().numpy()
    
    # Normalize to 0-1
    min_val = patch_attn.min()
    max_val = patch_attn.max()
    if max_val - min_val > 1e-5:
        patch_attn = (patch_attn - min_val) / (max_val - min_val)
    else:
        patch_attn = np.zeros_like(patch_attn)
    
    # Ensure correct dtype and contiguous array for cv2
    patch_attn = np.ascontiguousarray(patch_attn, dtype=np.float32)
    
    # Process input image
    if input_image is not None:
        # Get image dimensions
        img_h, img_w = input_image.shape[2], input_image.shape[3]
        
        # Convert image tensor to numpy (H, W, C)
        img_np = input_image[0].permute(1, 2, 0).detach().cpu().numpy()
        
        # Normalize image to 0-1 range
        if img_np.min() < 0:  # Assuming [-1, 1]
            img_np = (img_np + 1.0) / 2.0
        elif img_np.max() > 1:  # Assuming [0, 255]
            img_np = img_np / 255.0
        
        # Clip values to ensure valid range
        img_np = np.clip(img_np, 0, 1)
        
        # Resize attention map to match input image size
        heatmap = cv2.resize(patch_attn, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        
        # Create overlay visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(img_np)
        plt.imshow(heatmap, cmap='jet', alpha=alpha)
        plt.axis('off')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        
    else:
        # Just save the heatmap
        plt.figure(figsize=(8, 8))
        plt.imshow(patch_attn, cmap='jet')
        plt.axis('off')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()