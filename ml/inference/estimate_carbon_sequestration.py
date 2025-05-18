import os
import sys
import logging
import torch
import numpy as np
import json
from datetime import datetime

# Add parent directory to path to import from other modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from inference.predict_forest_change import ForestChangePredictor
from utils.data_preparation import calculate_ndvi

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CarbonSequestrationEstimator:
    """Class for estimating carbon sequestration from forest cover change."""
    
    def __init__(self, forest_change_predictor=None):
        """
        Initialize the carbon sequestration estimator.
        
        Args:
            forest_change_predictor (ForestChangePredictor, optional): Predictor for forest cover change
        """
        if forest_change_predictor is None:
            self.predictor = ForestChangePredictor()
        else:
            self.predictor = forest_change_predictor
    
    def estimate_carbon_sequestration(self, before_image_path, after_image_path, output_dir):
        """
        Estimate carbon sequestration between two time points.
        
        Args:
            before_image_path (str): Path to directory containing Sentinel-2 bands for earlier time point
            after_image_path (str): Path to directory containing Sentinel-2 bands for later time point
            output_dir (str): Directory to save results
        
        Returns:
            dict: Carbon sequestration estimates and related data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Process before and after images
        before_results = self.predictor.process_satellite_image(
            before_image_path, 
            os.path.join(output_dir, 'before')
        )
        
        after_results = self.predictor.process_satellite_image(
            after_image_path, 
            os.path.join(output_dir, 'after')
        )
        
        # Calculate forest cover change between the two time points
        forest_loss_before = before_results['forest_loss_area_ha']
        forest_loss_after = after_results['forest_loss_area_ha']
        
        # Calculate net change
        # Positive value means net forest gain (carbon sequestration)
        # Negative value means net forest loss (carbon emissions)
        net_change_ha = forest_loss_before - forest_loss_after
        
        # Estimate carbon sequestration/emissions
        # Average carbon sequestration rate: ~5 tC/ha/year for tropical forests
        carbon_sequestration_rate = 5  # tC/ha/year
        co2_factor = 3.67  # tCO2/tC
        
        # Get time difference between images
        # In a real implementation, you would extract dates from image metadata
        # For this example, we'll assume a 1-year difference
        time_diff_years = 1
        
        # Calculate carbon sequestration
        carbon_sequestration = net_change_ha * carbon_sequestration_rate * time_diff_years  # tC
        co2_sequestration = carbon_sequestration * co2_factor  # tCO2
        
        # Prepare results
        results = {
            'before_image_path': before_image_path,
            'after_image_path': after_image_path,
            'before_results': before_results,
            'after_results': after_results,
            'net_forest_change_ha': float(net_change_ha),
            'time_difference_years': time_diff_years,
            'carbon_sequestration_tons': float(carbon_sequestration),
            'co2_sequestration_tons': float(co2_sequestration),
            'confidence_score': float((before_results['average_confidence'] + after_results['average_confidence']) / 2),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results as JSON
        with open(os.path.join(output_dir, 'carbon_sequestration_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualization
        self._create_visualization(results, os.path.join(output_dir, 'carbon_sequestration_visualization.png'))
        
        return results
    
    def _create_visualization(self, results, output_path):
        """Create a visualization of carbon sequestration results."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Load visualizations from before and after results
        before_viz = plt.imread(results['before_results']['visualization_path'])
        after_viz = plt.imread(results['after_results']['visualization_path'])
        
        # Plot before and after visualizations
        axes[0, 0].imshow(before_viz)
        axes[0, 0].set_title('Before: Forest Cover Change')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(after_viz)
        axes[0, 1].set_title('After: Forest Cover Change')
        axes[0, 1].axis('off')
        
        # Plot carbon sequestration results
        net_change = results['net_forest_change_ha']
        carbon_seq = results['carbon_sequestration_tons']
        co2_seq = results['co2_sequestration_tons']
        
        # Bar chart for net forest change
        bar_color = 'green' if net_change > 0 else 'red'
        axes[1, 0].bar(['Net Forest Change'], [net_change], color=bar_color)
        axes[1, 0].set_ylabel('Hectares')
        axes[1, 0].set_title('Net Forest Change')
        axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Bar chart for carbon and CO2
        carbon_color = 'green' if carbon_seq > 0 else 'red'
        co2_color = 'green' if co2_seq > 0 else 'red'
        
        axes[1, 1].bar(['Carbon', 'CO₂'], [carbon_seq, co2_seq], color=[carbon_color, co2_color])
        axes[1, 1].set_ylabel('Tons')
        axes[1, 1].set_title('Carbon & CO₂ Sequestration')
        axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add confidence score
        confidence = results['confidence_score'] * 100
        fig.text(0.5, 0.01, f'Confidence Score: {confidence:.1f}%', ha='center', fontsize=12)
        
        # Add legend explaining positive/negative values
        if net_change > 0:
            status_text = 'Net Forest Gain: Carbon Sequestration'
            status_color = 'green'
        else:
            status_text = 'Net Forest Loss: Carbon Emissions'
            status_color = 'red'
        
        status_patch = mpatches.Patch(color=status_color, label=status_text)
        fig.legend(handles=[status_patch], loc='upper center', bbox_to_anchor=(0.5, 0.05))
        
        # Save figure
        plt.tight_layout(rect=[0, 0.07, 1, 0.97])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to demonstrate carbon sequestration estimation."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Estimate carbon sequestration from forest cover change')
    parser.add_argument('--before_image', type=str, required=True, help='Path to directory containing Sentinel-2 bands for earlier time point')
    parser.add_argument('--after_image', type=str, required=True, help='Path to directory containing Sentinel-2 bands for later time point')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--model_path', type=str, help='Path to trained forest change model')
    args = parser.parse_args()
    
    # Initialize predictor and estimator
    predictor = ForestChangePredictor(model_path=args.model_path if args.model_path else None)
    estimator = CarbonSequestrationEstimator(predictor)
    
    # Estimate carbon sequestration
    results = estimator.estimate_carbon_sequestration(
        args.before_image,
        args.after_image,
        args.output_dir
    )
    
    # Print results
    logger.info(f"Net forest change: {results['net_forest_change_ha']:.2f} hectares")
    logger.info(f"Carbon sequestration: {results['carbon_sequestration_tons']:.2f} tons")
    logger.info(f"CO2 sequestration: {results['co2_sequestration_tons']:.2f} tons")
    logger.info(f"Confidence score: {results['confidence_score']:.2f}")
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
