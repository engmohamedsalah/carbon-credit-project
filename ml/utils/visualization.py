import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SatelliteImageViewer:
    """Class for visualizing satellite imagery and ML model results."""
    
    def __init__(self):
        """Initialize the satellite image viewer."""
        pass
    
    def create_rgb_composite(self, image_path, output_path=None, bands=['B04', 'B03', 'B02']):
        """
        Create RGB composite from Sentinel-2 bands.
        
        Args:
            image_path (str): Path to directory containing Sentinel-2 band files
            output_path (str, optional): Path to save RGB composite
            bands (list): List of bands to use for RGB composite (default: ['B04', 'B03', 'B02'])
        
        Returns:
            np.ndarray: RGB composite image
        """
        # Load bands
        rgb_bands = []
        for band in bands:
            band_path = os.path.join(image_path, f"{band}.tif")
            if not os.path.exists(band_path):
                logger.error(f"Band file {band_path} not found")
                return None
            
            try:
                import rasterio
                with rasterio.open(band_path) as src:
                    rgb_bands.append(src.read(1))
            except ImportError:
                logger.warning("rasterio not installed, falling back to PIL")
                try:
                    img = Image.open(band_path)
                    rgb_bands.append(np.array(img))
                except Exception as e:
                    logger.error(f"Error loading band {band}: {str(e)}")
                    return None
        
        # Stack bands to create RGB image
        rgb_image = np.stack(rgb_bands, axis=2)
        
        # Normalize each band to [0, 1]
        for i in range(3):
            band_min, band_max = rgb_image[:,:,i].min(), rgb_image[:,:,i].max()
            rgb_image[:,:,i] = np.clip((rgb_image[:,:,i] - band_min) / (band_max - band_min), 0, 1)
        
        # Convert to 8-bit
        rgb_image = (rgb_image * 255).astype(np.uint8)
        
        # Save RGB composite if output_path is provided
        if output_path:
            plt.figure(figsize=(10, 10))
            plt.imshow(rgb_image)
            plt.axis('off')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"RGB composite saved to {output_path}")
        
        return rgb_image
    
    def create_ndvi_visualization(self, image_path, output_path=None):
        """
        Create NDVI visualization from Sentinel-2 bands.
        
        Args:
            image_path (str): Path to directory containing Sentinel-2 band files
            output_path (str, optional): Path to save NDVI visualization
        
        Returns:
            np.ndarray: NDVI image
        """
        # Load red and NIR bands
        red_path = os.path.join(image_path, "B04.tif")
        nir_path = os.path.join(image_path, "B08.tif")
        
        if not os.path.exists(red_path) or not os.path.exists(nir_path):
            logger.error(f"Required band files not found")
            return None
        
        try:
            import rasterio
            with rasterio.open(red_path) as src_red, rasterio.open(nir_path) as src_nir:
                red = src_red.read(1)
                nir = src_nir.read(1)
        except ImportError:
            logger.warning("rasterio not installed, falling back to PIL")
            try:
                red = np.array(Image.open(red_path))
                nir = np.array(Image.open(nir_path))
            except Exception as e:
                logger.error(f"Error loading bands: {str(e)}")
                return None
        
        # Calculate NDVI
        ndvi = (nir - red) / (nir + red + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Create colormap for NDVI
        if output_path:
            plt.figure(figsize=(10, 10))
            plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
            plt.colorbar(label='NDVI')
            plt.title('Normalized Difference Vegetation Index (NDVI)')
            plt.axis('off')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"NDVI visualization saved to {output_path}")
        
        return ndvi
    
    def visualize_forest_change(self, prediction_path, rgb_image_path, output_path=None):
        """
        Visualize forest change prediction.
        
        Args:
            prediction_path (str): Path to forest change prediction file
            rgb_image_path (str): Path to RGB composite image or directory with Sentinel-2 bands
            output_path (str, optional): Path to save visualization
        
        Returns:
            np.ndarray: Visualization image
        """
        # Load prediction
        try:
            import rasterio
            with rasterio.open(prediction_path) as src:
                prediction = src.read(1)
        except ImportError:
            logger.warning("rasterio not installed, falling back to PIL")
            try:
                prediction = np.array(Image.open(prediction_path))
            except Exception as e:
                logger.error(f"Error loading prediction: {str(e)}")
                return None
        
        # Load or create RGB image
        if os.path.isdir(rgb_image_path):
            rgb_image = self.create_rgb_composite(rgb_image_path)
        else:
            try:
                rgb_image = np.array(Image.open(rgb_image_path))
            except Exception as e:
                logger.error(f"Error loading RGB image: {str(e)}")
                return None
        
        # Create a color overlay for predictions (red for forest loss)
        overlay = np.zeros_like(rgb_image)
        overlay[prediction > 0, 0] = 255  # Red channel
        
        # Blend RGB image with prediction overlay
        alpha = 0.5
        import cv2
        blended = cv2.addWeighted(rgb_image, 1 - alpha, overlay, alpha, 0)
        
        # Save visualization if output_path is provided
        if output_path:
            plt.figure(figsize=(10, 10))
            plt.imshow(blended)
            plt.title('Forest Change Detection')
            plt.axis('off')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Forest change visualization saved to {output_path}")
        
        return blended
    
    def visualize_explanation(self, explanation_path, rgb_image_path, output_path=None):
        """
        Visualize model explanation.
        
        Args:
            explanation_path (str): Path to explanation file
            rgb_image_path (str): Path to RGB composite image or directory with Sentinel-2 bands
            output_path (str, optional): Path to save visualization
        
        Returns:
            np.ndarray: Visualization image
        """
        # Load explanation
        try:
            import rasterio
            with rasterio.open(explanation_path) as src:
                explanation = src.read(1)
        except ImportError:
            logger.warning("rasterio not installed, falling back to PIL")
            try:
                explanation = np.array(Image.open(explanation_path))
            except Exception as e:
                logger.error(f"Error loading explanation: {str(e)}")
                return None
        
        # Load or create RGB image
        if os.path.isdir(rgb_image_path):
            rgb_image = self.create_rgb_composite(rgb_image_path)
        else:
            try:
                rgb_image = np.array(Image.open(rgb_image_path))
            except Exception as e:
                logger.error(f"Error loading RGB image: {str(e)}")
                return None
        
        # Normalize explanation to [0, 1]
        explanation_normalized = (explanation - explanation.min()) / (explanation.max() - explanation.min() + 1e-8)
        
        # Create a heatmap for explanations
        import cv2
        heatmap = cv2.applyColorMap((explanation_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Save visualization if output_path is provided
        if output_path:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(1, 2, 1)
            plt.imshow(rgb_image)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(heatmap)
            plt.title('Explanation (Feature Importance)')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Explanation visualization saved to {output_path}")
        
        return heatmap

def main():
    """Main function to demonstrate the satellite image viewer."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Visualize satellite imagery and ML model results')
    parser.add_argument('--image_path', type=str, required=True, help='Path to directory containing Sentinel-2 band files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save visualizations')
    parser.add_argument('--prediction_path', type=str, help='Path to forest change prediction file')
    parser.add_argument('--explanation_path', type=str, help='Path to explanation file')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize viewer
    viewer = SatelliteImageViewer()
    
    # Create RGB composite
    rgb_output_path = os.path.join(args.output_dir, 'rgb_composite.png')
    viewer.create_rgb_composite(args.image_path, rgb_output_path)
    
    # Create NDVI visualization
    ndvi_output_path = os.path.join(args.output_dir, 'ndvi.png')
    viewer.create_ndvi_visualization(args.image_path, ndvi_output_path)
    
    # Visualize forest change if prediction_path is provided
    if args.prediction_path:
        change_output_path = os.path.join(args.output_dir, 'forest_change.png')
        viewer.visualize_forest_change(args.prediction_path, args.image_path, change_output_path)
    
    # Visualize explanation if explanation_path is provided
    if args.explanation_path:
        explanation_output_path = os.path.join(args.output_dir, 'explanation.png')
        viewer.visualize_explanation(args.explanation_path, args.image_path, explanation_output_path)
    
    logger.info(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
