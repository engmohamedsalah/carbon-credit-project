"""
Mock Satellite Imagery Processing Module

This module provides mock functionality for satellite imagery acquisition and processing
for carbon credit verification in Phase 2 of the project.
"""

import json
import random
import numpy as np
from datetime import datetime
from PIL import Image

class SatelliteImageProcessor:
    """Mock class for satellite imagery processing"""
    
    def __init__(self):
        self.providers = ["Sentinel-2", "Landsat-8", "PlanetScope"]
        self.resolutions = {"Sentinel-2": "10m", "Landsat-8": "30m", "PlanetScope": "3m"}
    
    def acquire_imagery(self, location: str):
        """Mock method to acquire satellite imagery for a location"""
        provider = random.choice(self.providers)
        resolution = self.resolutions[provider]
        
        # Simulating a delay in acquiring imagery
        acquisition_date = datetime.now().isoformat()
        
        return {
            "provider": provider,
            "resolution": resolution,
            "location": location,
            "acquisition_date": acquisition_date,
            "status": "Acquired"
        }
    
    def process_imagery(self, location: str):
        """Process satellite imagery for land cover classification"""
        # Mock land cover classes
        land_cover_classes = [
            "Dense Forest", "Sparse Forest", "Grassland", 
            "Cropland", "Barren", "Water", "Urban"
        ]
        
        # Generate mock land cover data
        total_area = random.uniform(50, 500)  # hectares
        
        land_cover_data = {}
        remaining_area = total_area
        
        for cls in land_cover_classes[:-1]:
            if remaining_area <= 0:
                land_cover_data[cls] = 0
            else:
                area = random.uniform(0, remaining_area * 0.5)
                land_cover_data[cls] = round(area, 2)
                remaining_area -= area
        
        land_cover_data[land_cover_classes[-1]] = round(remaining_area, 2)
        
        # Generate mock carbon estimation
        carbon_density = {
            "Dense Forest": random.uniform(200, 350),      # tC/ha
            "Sparse Forest": random.uniform(100, 200),     # tC/ha
            "Grassland": random.uniform(10, 50),          # tC/ha
            "Cropland": random.uniform(5, 20),            # tC/ha
            "Barren": random.uniform(0, 2),               # tC/ha
            "Water": 0,                                    # tC/ha
            "Urban": random.uniform(0, 5)                  # tC/ha
        }
        
        total_carbon = sum(land_cover_data[cls] * carbon_density[cls] for cls in land_cover_classes)
        
        return {
            "location": location,
            "total_area_hectares": round(total_area, 2),
            "land_cover_distribution": land_cover_data,
            "carbon_density": {cls: round(density, 2) for cls, density in carbon_density.items()},
            "total_carbon_estimate_tonnes": round(total_carbon, 2),
            "confidence_level": random.uniform(0.75, 0.95),
            "processing_date": datetime.now().isoformat()
        }
    
    def generate_mock_image(self, width=500, height=500):
        """Generate a mock satellite image for visualization"""
        # Create a random land cover image
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Forest (green)
        forest_mask = np.random.random((height, width)) > 0.7
        image[forest_mask] = [34, 139, 34]
        
        # Water (blue)
        water_mask = np.random.random((height, width)) > 0.9
        image[water_mask] = [65, 105, 225]
        
        # Cropland (light green)
        crop_mask = np.random.random((height, width)) > 0.8
        image[crop_mask & ~forest_mask & ~water_mask] = [144, 238, 144]
        
        # Urban (gray)
        urban_mask = np.random.random((height, width)) > 0.9
        image[urban_mask & ~forest_mask & ~water_mask & ~crop_mask] = [128, 128, 128]
        
        # Grassland (yellow-green)
        grass_mask = ~forest_mask & ~water_mask & ~crop_mask & ~urban_mask
        image[grass_mask] = [154, 205, 50]
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        return pil_image

# Example usage
if __name__ == "__main__":
    processor = SatelliteImageProcessor()
    location = "10.5,20.3"  # Latitude, Longitude
    
    imagery_info = processor.acquire_imagery(location)
    print(json.dumps(imagery_info, indent=2))
    
    processing_results = processor.process_imagery(location)
    print(json.dumps(processing_results, indent=2))
    
    # Generate and save a mock image
    mock_image = processor.generate_mock_image()
    mock_image.save("mock_satellite_image.png")
    print("Mock satellite image saved as mock_satellite_image.png") 