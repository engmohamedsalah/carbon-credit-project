import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XAIVisualizer:
    """Class for creating explainable AI visualizations for the forest change model."""
    
    def __init__(self):
        """Initialize the XAI visualizer."""
        pass
    
    def create_feature_importance_heatmap(self, attribution_map, rgb_image, output_path=None):
        """
        Create a feature importance heatmap overlay on the original image.
        
        Args:
            attribution_map (np.ndarray): Attribution map from explainable AI method
            rgb_image (np.ndarray): Original RGB image
            output_path (str, optional): Path to save visualization
        
        Returns:
            np.ndarray: Heatmap visualization
        """
        # Normalize attribution map to [-1, 1]
        attribution_max = np.abs(attribution_map).max()
        if attribution_max > 0:
            attribution_map = attribution_map / attribution_max
        
        # Create a heatmap using a diverging colormap (blue-white-red)
        # Blue for negative attribution, red for positive attribution
        import matplotlib.cm as cm
        cmap = cm.get_cmap('coolwarm')
        
        # Scale to [0, 1] for colormap
        attribution_scaled = (attribution_map + 1) / 2
        heatmap = cmap(attribution_scaled)[:, :, :3]  # Remove alpha channel
        
        # Convert to 8-bit
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Create overlay with transparency
        alpha = 0.7
        overlay = np.zeros_like(rgb_image)
        for i in range(3):
            overlay[:, :, i] = (1 - alpha) * rgb_image[:, :, i] + alpha * heatmap[:, :, i]
        
        # Save visualization if output_path is provided
        if output_path:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(1, 3, 1)
            plt.imshow(rgb_image)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(heatmap)
            plt.title('Feature Importance')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title('Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature importance heatmap saved to {output_path}")
        
        return overlay
    
    def create_saliency_map(self, attribution_map, rgb_image, output_path=None):
        """
        Create a saliency map highlighting the most important regions.
        
        Args:
            attribution_map (np.ndarray): Attribution map from explainable AI method
            rgb_image (np.ndarray): Original RGB image
            output_path (str, optional): Path to save visualization
        
        Returns:
            np.ndarray: Saliency map visualization
        """
        # Take absolute value to focus on magnitude of importance
        saliency = np.abs(attribution_map)
        
        # Normalize to [0, 1]
        saliency = saliency / saliency.max() if saliency.max() > 0 else saliency
        
        # Create a heatmap using a sequential colormap (white to red)
        import matplotlib.cm as cm
        cmap = cm.get_cmap('hot')
        
        heatmap = cmap(saliency)[:, :, :3]  # Remove alpha channel
        
        # Convert to 8-bit
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Create overlay with transparency
        alpha = 0.7
        overlay = np.zeros_like(rgb_image)
        for i in range(3):
            overlay[:, :, i] = (1 - alpha) * rgb_image[:, :, i] + alpha * heatmap[:, :, i]
        
        # Save visualization if output_path is provided
        if output_path:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(1, 3, 1)
            plt.imshow(rgb_image)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(heatmap)
            plt.title('Saliency Map')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title('Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saliency map saved to {output_path}")
        
        return overlay
    
    def create_comparison_visualization(self, before_image, after_image, prediction, explanation, output_path=None):
        """
        Create a comprehensive visualization comparing before/after images with prediction and explanation.
        
        Args:
            before_image (np.ndarray): RGB image from earlier time point
            after_image (np.ndarray): RGB image from later time point
            prediction (np.ndarray): Forest change prediction mask
            explanation (np.ndarray): Attribution map from explainable AI method
            output_path (str, optional): Path to save visualization
        
        Returns:
            np.ndarray: Comprehensive visualization
        """
        # Create prediction overlay
        overlay = np.copy(after_image)
        overlay[prediction > 0, 0] = 255  # Red for forest loss
        overlay[prediction > 0, 1:] = 0
        
        # Create explanation heatmap
        explanation_max = np.abs(explanation).max()
        if explanation_max > 0:
            explanation = explanation / explanation_max
        
        import matplotlib.cm as cm
        cmap = cm.get_cmap('coolwarm')
        explanation_scaled = (explanation + 1) / 2
        heatmap = cmap(explanation_scaled)[:, :, :3]  # Remove alpha channel
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Save visualization if output_path is provided
        if output_path:
            plt.figure(figsize=(20, 15))
            
            plt.subplot(2, 2, 1)
            plt.imshow(before_image)
            plt.title('Before Image')
            plt.axis('off')
            
            plt.subplot(2, 2, 2)
            plt.imshow(after_image)
            plt.title('After Image')
            plt.axis('off')
            
            plt.subplot(2, 2, 3)
            plt.imshow(overlay)
            plt.title('Forest Change Prediction')
            plt.axis('off')
            
            plt.subplot(2, 2, 4)
            plt.imshow(heatmap)
            plt.title('Explanation (Feature Importance)')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Comparison visualization saved to {output_path}")
        
        # Create a combined visualization
        combined = np.zeros((before_image.shape[0] * 2, before_image.shape[1] * 2, 3), dtype=np.uint8)
        combined[:before_image.shape[0], :before_image.shape[1]] = before_image
        combined[:before_image.shape[0], before_image.shape[1]:] = after_image
        combined[before_image.shape[0]:, :before_image.shape[1]] = overlay
        combined[before_image.shape[0]:, before_image.shape[1]:] = heatmap
        
        return combined
    
    def create_confidence_visualization(self, prediction, confidence, rgb_image, output_path=None):
        """
        Create a visualization showing prediction confidence.
        
        Args:
            prediction (np.ndarray): Forest change prediction mask
            confidence (np.ndarray): Confidence scores for each pixel
            rgb_image (np.ndarray): Original RGB image
            output_path (str, optional): Path to save visualization
        
        Returns:
            np.ndarray: Confidence visualization
        """
        # Create a confidence heatmap only for predicted forest loss areas
        confidence_map = np.zeros_like(confidence)
        confidence_map[prediction > 0] = confidence[prediction > 0]
        
        # Normalize to [0, 1]
        confidence_max = confidence_map.max()
        if confidence_max > 0:
            confidence_map = confidence_map / confidence_max
        
        # Create a heatmap using a sequential colormap (white to green)
        import matplotlib.cm as cm
        cmap = cm.get_cmap('Greens')
        
        heatmap = cmap(confidence_map)[:, :, :3]  # Remove alpha channel
        
        # Convert to 8-bit
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Create overlay with transparency
        alpha = 0.7
        overlay = np.zeros_like(rgb_image)
        for i in range(3):
            overlay[:, :, i] = (1 - alpha) * rgb_image[:, :, i] + alpha * heatmap[:, :, i]
        
        # Save visualization if output_path is provided
        if output_path:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(1, 3, 1)
            plt.imshow(rgb_image)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(heatmap)
            plt.title('Prediction Confidence')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title('Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confidence visualization saved to {output_path}")
        
        return overlay

def main():
    """Main function to demonstrate the XAI visualizer."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Create explainable AI visualizations')
    parser.add_argument('--rgb_image', type=str, required=True, help='Path to RGB image')
    parser.add_argument('--attribution_map', type=str, required=True, help='Path to attribution map')
    parser.add_argument('--prediction', type=str, required=True, help='Path to prediction mask')
    parser.add_argument('--confidence', type=str, help='Path to confidence map')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save visualizations')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load images
    try:
        import rasterio
        with rasterio.open(args.rgb_image) as src:
            rgb_image = src.read()
            # Transpose to (height, width, channels)
            rgb_image = np.transpose(rgb_image, (1, 2, 0))
        
        with rasterio.open(args.attribution_map) as src:
            attribution_map = src.read(1)
        
        with rasterio.open(args.prediction) as src:
            prediction = src.read(1)
        
        if args.confidence:
            with rasterio.open(args.confidence) as src:
                confidence = src.read(1)
        else:
            confidence = None
    except ImportError:
        logger.warning("rasterio not installed, falling back to PIL")
        try:
            rgb_image = np.array(Image.open(args.rgb_image))
            attribution_map = np.array(Image.open(args.attribution_map))
            prediction = np.array(Image.open(args.prediction))
            if args.confidence:
                confidence = np.array(Image.open(args.confidence))
            else:
                confidence = None
        except Exception as e:
            logger.error(f"Error loading images: {str(e)}")
            return
    
    # Initialize visualizer
    visualizer = XAIVisualizer()
    
    # Create feature importance heatmap
    heatmap_path = os.path.join(args.output_dir, 'feature_importance.png')
    visualizer.create_feature_importance_heatmap(attribution_map, rgb_image, heatmap_path)
    
    # Create saliency map
    saliency_path = os.path.join(args.output_dir, 'saliency_map.png')
    visualizer.create_saliency_map(attribution_map, rgb_image, saliency_path)
    
    # Create confidence visualization if confidence map is provided
    if confidence is not None:
        confidence_path = os.path.join(args.output_dir, 'confidence.png')
        visualizer.create_confidence_visualization(prediction, confidence, rgb_image, confidence_path)
    
    logger.info(f"XAI visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
