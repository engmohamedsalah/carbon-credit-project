import React, { useEffect, useRef } from 'react';
import { Box } from '@mui/material';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet-draw/dist/leaflet.draw.css';
import 'leaflet-draw';
import 'leaflet-geometryutil';

// Fix Leaflet icon issues
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const MapComponent = ({ onGeometryChange, onAreaCalculated, initialGeometry = null }) => {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const drawnItemsRef = useRef(null);

  // Initialize map only once
  useEffect(() => {
    if (!mapInstanceRef.current) {
      const map = L.map(mapRef.current).setView([0, 0], 2);
      
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      }).addTo(map);
      
      // Initialize FeatureGroup for drawn items
      const drawnItems = new L.FeatureGroup();
      map.addLayer(drawnItems);
      
      // Initialize draw control
      const drawControl = new L.Control.Draw({
        edit: {
          featureGroup: drawnItems,
          poly: {
            allowIntersection: false
          }
        },
        draw: {
          polygon: {
            allowIntersection: false,
            showArea: true
          },
          polyline: false,
          circle: false,
          circlemarker: false,
          marker: false,
          rectangle: true
        }
      });
      map.addControl(drawControl);
      
      // Event handler for when a shape is created
      map.on(L.Draw.Event.CREATED, (event) => {
        const layer = event.layer;
        // Clear previous drawings before adding new one
        drawnItems.clearLayers();
        drawnItems.addLayer(layer);
        
        // Convert to GeoJSON and pass to parent
        const geoJSON = layer.toGeoJSON();
        if (onGeometryChange) {
          onGeometryChange(geoJSON.geometry);
        }
        
        // Calculate area if it's a polygon
        if (layer instanceof L.Polygon || layer instanceof L.Rectangle) {
          const area = L.GeometryUtil.geodesicArea(layer.getLatLngs()[0]);
          const areaInHectares = (area / 10000).toFixed(2); // Convert m² to hectares
          if (onAreaCalculated) {
            onAreaCalculated(areaInHectares);
          }
        }
      });
      
      // Event handler for when a shape is edited
      map.on(L.Draw.Event.EDITED, (event) => {
        const layers = event.layers;
        layers.eachLayer((layer) => {
          const geoJSON = layer.toGeoJSON();
          if (onGeometryChange) {
            onGeometryChange(geoJSON.geometry);
          }
          
          // Recalculate area if it's a polygon
          if (layer instanceof L.Polygon || layer instanceof L.Rectangle) {
            const area = L.GeometryUtil.geodesicArea(layer.getLatLngs()[0]);
            const areaInHectares = (area / 10000).toFixed(2);
            if (onAreaCalculated) {
              onAreaCalculated(areaInHectares);
            }
          }
        });
      });
      
      // Event handler for when a shape is deleted
      map.on(L.Draw.Event.DELETED, () => {
        if (drawnItems.getLayers().length === 0) {
          if (onGeometryChange) {
            onGeometryChange(null);
          }
        }
      });
      
      mapInstanceRef.current = map;
      drawnItemsRef.current = drawnItems;
      
      // Load initial geometry if provided
      if (initialGeometry && drawnItems) {
        const layer = L.geoJSON(initialGeometry);
        layer.eachLayer((l) => {
          drawnItems.addLayer(l);
        });
        // Fit map to the geometry bounds
        map.fitBounds(layer.getBounds());
      }
    }
    
    // Cleanup function
    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.off();
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []); // Empty dependency array - initialize only once
  
  // Handle initialGeometry changes in a separate effect
  useEffect(() => {
    if (initialGeometry && drawnItemsRef.current && mapInstanceRef.current) {
      drawnItemsRef.current.clearLayers();
      
      const layer = L.geoJSON(initialGeometry);
      layer.eachLayer((l) => {
        drawnItemsRef.current.addLayer(l);
      });
      
      // Fit map to the geometry bounds
      mapInstanceRef.current.fitBounds(layer.getBounds());
    }
  }, [initialGeometry]);
  
  return (
    <Box 
      ref={mapRef} 
      sx={{ 
        width: '100%', 
        height: '100%',
        '& .leaflet-container': {
          height: '100%',
          width: '100%',
        }
      }} 
    />
  );
};

export default MapComponent;
