import os
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from PIL import Image
import logging
import sys
import json
from captum.attr import IntegratedGradients, GradientShap, Occlusion
import cv2

# Add parent directory to path to import from training
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from training.train_forest_change import UNet, calculate_ndvi, PATCH_SIZE

# Sentinel-2 bands to use for inference
SENTINEL_BANDS = ['B02', 'B03', 'B04', 'B08']

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'forest_change_unet.pth')

class ForestChangePredictor:
    """Class for predicting forest cover change using trained model."""
    
    def __init__(self, model_path=MODEL_PATH):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path (str): Path to the trained model weights
        """
        self.model = UNet(in_channels=4, out_channels=2).to(DEVICE)
        
        # Load model weights if they exist
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}. Using untrained model.")
        
        self.model.eval()
        
        # Initialize explainers
        self.integrated_gradients = IntegratedGradients(self.model)
        self.gradient_shap = GradientShap(self.model)
        self.occlusion = Occlusion(self.model)
    
    def preprocess_image(self, image_path):
        """
        Preprocess Sentinel-2 imagery for prediction.
        
        Args:
            image_path (str): Path to directory containing Sentinel-2 band files
        
        Returns:
            torch.Tensor: Preprocessed image tensor ready for model input
        """
        # Load bands
        bands = []
        for band in SENTINEL_BANDS:
            band_path = os.path.join(image_path, f"{band}.jp2")
            with rasterio.open(band_path) as src:
                band_data = src.read(1)
                bands.append(band_data)
        
        # Stack bands
        image = np.stack(bands, axis=0)
        
        # Normalize bands to [0, 1]
        for i in range(len(bands)):
            band_min, band_max = image[i].min(), image[i].max()
            image[i] = (image[i] - band_min) / (band_max - band_min + 1e-8)
        
        # Convert to PyTorch tensor
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # Add batch dimension
        
        return image_tensor
    
    def predict(self, image_tensor):
        """
        Predict forest cover change from preprocessed image tensor.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor
        
        Returns:
            np.ndarray: Prediction mask (0: no change, 1: forest loss)
            float: Confidence score
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(DEVICE)
            output = self.model(image_tensor)
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
            # Get predicted class (0: no change, 1: forest loss)
            _, predicted = torch.max(probabilities, 1)
            
            # Convert to numpy arrays
            predicted = predicted.cpu().numpy()[0]  # Remove batch dimension
            probabilities = probabilities.cpu().numpy()[0]  # Remove batch dimension
            
            # Calculate confidence score (mean probability of predicted class)
            confidence = np.mean(np.max(probabilities, axis=0))
            
            return predicted, confidence
    
    def explain_prediction(self, image_tensor, method='integrated_gradients'):
        """
        Generate explanation for model prediction.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor
            method (str): Explanation method ('integrated_gradients', 'gradient_shap', or 'occlusion')
        
        Returns:
            np.ndarray: Attribution map highlighting important regions for prediction
        """
        image_tensor = image_tensor.to(DEVICE)
        baseline = torch.zeros_like(image_tensor).to(DEVICE)

        def custom_forward(x):
            output = self.model(x)
            return output[:, 1, :, :].mean(dim=(1, 2))

        if method == 'integrated_gradients':
            integrated_gradients = IntegratedGradients(custom_forward)
            attributions = integrated_gradients.attribute(image_tensor, baseline, target=None)
        elif method == 'gradient_shap':
            gradient_shap = GradientShap(custom_forward)
            attributions = gradient_shap.attribute(image_tensor, baseline, target=None)
        elif method == 'occlusion':
            occlusion = Occlusion(custom_forward)
            attributions = occlusion.attribute(image_tensor,
                                               sliding_window_shapes=(1, 16, 16),
                                               strides=(1, 8, 8),
                                               target=None)
        else:
            raise ValueError(f"Unknown explanation method: {method}")
        
        # Sum attributions across channels
        attribution_map = torch.sum(attributions, dim=1).squeeze(0).cpu().numpy()
        
        # Normalize to [-1, 1]
        attribution_max = np.abs(attribution_map).max()
        if attribution_max > 0:
            attribution_map = attribution_map / attribution_max
        
        return attribution_map
    
    def process_satellite_image(self, image_path, output_dir, patch_size=PATCH_SIZE, overlap=64):
        """
        Process a full satellite image by dividing it into patches, making predictions,
        and stitching the results back together.
        
        Args:
            image_path (str): Path to directory containing Sentinel-2 band files
            output_dir (str): Directory to save results
            patch_size (int): Size of patches to process
            overlap (int): Overlap between patches
        
        Returns:
            dict: Results including prediction path, confidence, and explanation path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get image dimensions from first band
        with rasterio.open(os.path.join(image_path, f"{SENTINEL_BANDS[0]}.jp2")) as src:
            height, width = src.height, src.width
            profile = src.profile
        print(f"[DEBUG] Image size: {height}x{width}")
        print(f"[DEBUG] Patch size: {patch_size}, Overlap: {overlap}")
        stride = patch_size - overlap
        n_patches_y = (height - patch_size) // stride + 1
        n_patches_x = (width - patch_size) // stride + 1
        print(f"[DEBUG] Stride: {stride}, Expected patches: {n_patches_y} x {n_patches_x} = {n_patches_y * n_patches_x}")
        
        # Initialize full-size prediction and confidence maps
        prediction_map = np.zeros((height, width), dtype=np.float32)
        confidence_map = np.zeros((height, width), dtype=np.float32)
        explanation_map = np.zeros((height, width), dtype=np.float32)
        
        # Process image in patches
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                print(f"[DEBUG] Processing patch at (x={x}, y={y})")
                # Extract patch
                patch_bands = []
                for band in SENTINEL_BANDS:
                    with rasterio.open(os.path.join(image_path, f"{band}.jp2")) as src:
                        window = Window(x, y, patch_size, patch_size)
                        patch_bands.append(src.read(1, window=window))
                # Stack and normalize patch
                patch = np.stack(patch_bands, axis=0)
                for i in range(len(patch_bands)):
                    band_min, band_max = patch[i].min(), patch[i].max()
                    patch[i] = (patch[i] - band_min) / (band_max - band_min + 1e-8)
                # Convert to tensor
                patch_tensor = torch.from_numpy(patch).float().unsqueeze(0)
                # Make prediction
                patch_prediction, patch_confidence = self.predict(patch_tensor)
                # Generate explanation (SKIPPED FOR SPEED)
                # patch_explanation = self.explain_prediction(patch_tensor)
                # Create weight mask for blending (higher weight in center, lower at edges)
                y_grid, x_grid = np.mgrid[0:patch_size, 0:patch_size]
                center_y, center_x = patch_size // 2, patch_size // 2
                dist_from_center = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
                weight_mask = np.clip(1 - dist_from_center / (patch_size // 2), 0, 1)
                # Update prediction, confidence, and explanation maps with weighted values
                prediction_map[y:y+patch_size, x:x+patch_size] += patch_prediction * weight_mask
                confidence_map[y:y+patch_size, x:x+patch_size] += patch_confidence * weight_mask
                # explanation_map[y:y+patch_size, x:x+patch_size] += patch_explanation * weight_mask
        
        # Normalize maps
        prediction_map = (prediction_map > 0.5).astype(np.uint8)
        
        # Save results
        prediction_path = os.path.join(output_dir, 'forest_change_prediction.tif')
        confidence_path = os.path.join(output_dir, 'confidence.tif')
        # explanation_path = os.path.join(output_dir, 'explanation.tif')
        visualization_path = os.path.join(output_dir, 'visualization.png')
        print(f"[DEBUG] Writing prediction map to {prediction_path}")
        # Update profile for output files
        profile.update(count=1, dtype=rasterio.uint8)
        
        # Save prediction map
        with rasterio.open(prediction_path, 'w', driver='GTiff', **profile) as dst:
            dst.write(prediction_map, 1)
        print(f"[DEBUG] Prediction map written.")
        
        # Update profile for floating point data
        profile.update(dtype=rasterio.float32)
        print(f"[DEBUG] Writing confidence map to {confidence_path}")
        
        # Save confidence map
        with rasterio.open(confidence_path, 'w', driver='GTiff', **profile) as dst:
            dst.write(confidence_map, 1)
        print(f"[DEBUG] Confidence map written.")
        
        # Save explanation map (SKIPPED)
        # print(f"[DEBUG] Writing explanation map to {explanation_path}")
        # with rasterio.open(explanation_path, 'w', **profile) as dst:
        #     dst.write(explanation_map, 1)
        # print(f"[DEBUG] Explanation map written.")
        
        print(f"[DEBUG] Writing visualization to {visualization_path}")
        # Create visualization (skip explanation)
        self._create_visualization(image_path, prediction_map, None, visualization_path)
        print(f"[DEBUG] Visualization written.")
        
        # Calculate statistics
        forest_loss_area = np.sum(prediction_map) * 10 * 10 / 10000  # Convert pixels to hectares (assuming 10m resolution)
        average_confidence = np.mean(confidence_map[prediction_map > 0]) if np.any(prediction_map > 0) else 0
        
        # Prepare results
        results = {
            'prediction_path': prediction_path,
            'confidence_path': confidence_path,
            # 'explanation_path': explanation_path,
            'visualization_path': visualization_path,
            'forest_loss_area_ha': float(forest_loss_area),
            'average_confidence': float(average_confidence),
            'carbon_impact': self._estimate_carbon_impact(forest_loss_area)
        }
        
        # Save results as JSON
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _create_visualization(self, image_path, prediction_map, explanation_map, output_path):
        """Create a visualization of the prediction and explanation."""
        # Load RGB bands
        rgb_bands = []
        for band in ['B04', 'B03', 'B02']:
            with rasterio.open(os.path.join(image_path, f"{band}.jp2")) as src:
                rgb_bands.append(src.read(1))
        
        # Stack and normalize RGB image
        rgb_image = np.stack(rgb_bands, axis=2)
        for i in range(3):
            band_min, band_max = rgb_image[:,:,i].min(), rgb_image[:,:,i].max()
            rgb_image[:,:,i] = np.clip((rgb_image[:,:,i] - band_min) / (band_max - band_min), 0, 1)
        
        # Convert to 8-bit
        rgb_image = (rgb_image * 255).astype(np.uint8)
        
        # Create a color overlay for predictions (red for forest loss)
        overlay = np.zeros_like(rgb_image)
        overlay[prediction_map > 0, 0] = 255  # Red channel
        
        # Create a heatmap for explanations
        explanation_normalized = (explanation_map - explanation_map.min()) / (explanation_map.max() - explanation_map.min() + 1e-8)
        heatmap = cv2.applyColorMap((explanation_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend RGB image with prediction overlay
        alpha = 0.5
        blended = cv2.addWeighted(rgb_image, 1 - alpha, overlay, alpha, 0)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot RGB image
        axes[0].imshow(rgb_image)
        axes[0].set_title('Sentinel-2 RGB')
        axes[0].axis('off')
        
        # Plot prediction overlay
        axes[1].imshow(blended)
        axes[1].set_title('Forest Loss Prediction')
        axes[1].axis('off')
        
        # Plot explanation heatmap (SKIPPED)
        # axes[2].imshow(heatmap)
        # axes[2].set_title('Explanation (Feature Importance)')
        # axes[2].axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _estimate_carbon_impact(self, forest_loss_area_ha):
        """
        Estimate carbon impact from forest loss area.
        
        Args:
            forest_loss_area_ha (float): Forest loss area in hectares
        
        Returns:
            dict: Carbon impact estimates
        """
        # Average carbon density in tropical forests: ~150 tC/ha (tons of carbon per hectare)
        # Convert to CO2: multiply by 3.67 (ratio of molecular weights)
        carbon_density = 150  # tC/ha
        co2_factor = 3.67  # tCO2/tC
        
        carbon_loss = forest_loss_area_ha * carbon_density  # tC
        co2_emissions = carbon_loss * co2_factor  # tCO2
        
        return {
            'carbon_loss_tons': float(carbon_loss),
            'co2_emissions_tons': float(co2_emissions)
        }

def main():
    """Main function to demonstrate the forest change predictor."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Predict forest cover change from Sentinel-2 imagery')
    parser.add_argument('--image_path', type=str, required=True, help='Path to directory containing Sentinel-2 band files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Path to trained model')
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ForestChangePredictor(model_path=args.model_path)
    
    # Process image
    results = predictor.process_satellite_image(args.image_path, args.output_dir)
    
    # Print results
    logger.info(f"Forest loss area: {results['forest_loss_area_ha']:.2f} hectares")
    logger.info(f"Average confidence: {results['average_confidence']:.2f}")
    logger.info(f"Carbon loss: {results['carbon_impact']['carbon_loss_tons']:.2f} tons")
    logger.info(f"CO2 emissions: {results['carbon_impact']['co2_emissions_tons']:.2f} tons")
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
