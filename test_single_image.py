import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from model.segnet_mtl_cityscapes import SegNet
from model.mapfns import Mapfns
import argparse

def load_model(checkpoint_path, model_type='standard'):
    """Load the trained model from checkpoint"""
    # Initialize model
    model = SegNet(type_=model_type, class_nb=7).cuda()
    mapfns = Mapfns(tasks=['semantic', 'depth'], input_channels=[7, 1]).cuda()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Best performance: {checkpoint['best_performance']:.4f}")
    
    model.eval()
    return model, mapfns

def preprocess_image(image_path, target_size=(288, 384)):
    """
    Preprocess input image to match training format
    Args:
        image_path: path to input image
        target_size: (height, width) tuple
    """
    # Load image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path
    
    # Resize image
    image = image.resize((target_size[1], target_size[0]), Image.BILINEAR)
    
    # Convert to numpy array and normalize
    image_np = np.array(image).astype(np.float32) / 255.0
    
    # Convert to torch tensor and rearrange dimensions (H, W, C) -> (C, H, W)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor.cuda()

def postprocess_semantic(semantic_pred, num_classes=7):
    """
    Postprocess semantic segmentation prediction
    """
    # Get class predictions
    semantic_pred = torch.softmax(semantic_pred, dim=1)
    semantic_map = torch.argmax(semantic_pred, dim=1).squeeze().cpu().numpy()
    
    return semantic_map

def postprocess_depth(depth_pred):
    """
    Postprocess depth prediction
    """
    depth_map = depth_pred.squeeze().cpu().numpy()
    
    # Normalize depth for visualization
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    return depth_map, depth_normalized

def visualize_results(original_image, semantic_map, depth_map, save_path=None):
    """
    Visualize the results
    """
    # Define Cityscapes color palette for 7 classes
    # You may need to adjust these colors based on your specific class mapping
    colors = [
        [128, 64, 128],    # road
        [244, 35, 232],    # sidewalk  
        [70, 70, 70],      # building
        [102, 102, 156],   # wall
        [190, 153, 153],   # fence
        [153, 153, 153],   # pole
        [250, 170, 30],    # traffic light/sign
    ]
    
    # Create colored semantic map
    semantic_colored = np.zeros((semantic_map.shape[0], semantic_map.shape[1], 3), dtype=np.uint8)
    for class_id in range(len(colors)):
        mask = semantic_map == class_id
        semantic_colored[mask] = colors[class_id]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original image
    if isinstance(original_image, torch.Tensor):
        original_np = original_image.squeeze().permute(1, 2, 0).cpu().numpy()
    else:
        original_np = np.array(original_image) / 255.0 if np.max(original_image) > 1 else original_image
    
    axes[0, 0].imshow(original_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Semantic segmentation
    axes[0, 1].imshow(semantic_colored)
    axes[0, 1].set_title('Semantic Segmentation')
    axes[0, 1].axis('off')
    
    # Depth map
    axes[1, 0].imshow(depth_map, cmap='plasma')
    axes[1, 0].set_title('Depth Map')
    axes[1, 0].axis('off')
    
    # Overlay
    overlay = cv2.addWeighted(
        (original_np * 255).astype(np.uint8), 0.7,
        semantic_colored, 0.3, 0
    )
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Semantic Overlay')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to {save_path}")
    
    plt.show()

def test_single_image(image_path, checkpoint_path, model_type='standard', save_results=True):
    """
    Test the trained model on a single image
    """
    print("Loading model...")
    model, mapfns = load_model(checkpoint_path, model_type)
    
    print("Preprocessing image...")
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess_image(image_path)
    
    print("Running inference...")
    with torch.no_grad():
        # Forward pass
        predictions, logsigma, feat = model(input_tensor)
        semantic_pred, depth_pred = predictions
        
        print(f"Semantic prediction shape: {semantic_pred.shape}")
        print(f"Depth prediction shape: {depth_pred.shape}")
        
        # Postprocess predictions
        semantic_map = postprocess_semantic(semantic_pred)
        depth_map, depth_normalized = postprocess_depth(depth_pred)
        
        print("Visualizing results...")
        # Create save path
        save_path = None
        if save_results:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = f"results_{base_name}.png"
        
        # Visualize results
        visualize_results(input_tensor, semantic_map, depth_normalized, save_path)
        
        # Print some statistics
        print(f"\nResults Summary:")
        print(f"Unique semantic classes found: {np.unique(semantic_map)}")
        print(f"Depth range: {depth_map.min():.3f} to {depth_map.max():.3f}")
        
        return semantic_map, depth_map

def main():
    parser = argparse.ArgumentParser(description='Test trained multi-task model on single image')
    parser.add_argument('--image', required=True, type=str, help='Path to input image')
    parser.add_argument('--checkpoint', default='/content/mtl_xtc_onelabel_fixed_1.0_0.5_model_best.pth.tar', 
                       type=str, help='Path to model checkpoint')
    parser.add_argument('--model-type', default='standard', type=str, 
                       help='Model type: standard, wide, deep')
    parser.add_argument('--save', action='store_true', help='Save visualization results')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} not found!")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} not found!")
        return
    
    # Run inference
    try:
        semantic_map, depth_map = test_single_image(
            args.image, 
            args.checkpoint, 
            args.model_type,
            args.save
        )
        print("Inference completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

# Example usage:
# python test_single_image.py --image path/to/your/image.jpg --checkpoint /content/mtl_xtc_onelabel_fixed_1.0_0.5_model_best.pth.tar --save