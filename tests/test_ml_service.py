"""
Comprehensive Unit Tests for ML Service
Tests the machine learning integration service with proper mocking and error handling
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.ml_service import MLService, MLServiceError, MLServiceNotInitializedError, MLAnalysisError


class TestMLService:
    """Test suite for ML Service functionality"""

    @pytest.fixture
    def ml_service(self):
        """Create ML service instance for testing"""
        with patch('backend.services.ml_service.CarbonCreditVerificationPipeline') as mock_pipeline:
            mock_pipeline.return_value = Mock()
            service = MLService()
            service.is_initialized = True  # Force initialization for testing
            service.pipeline = Mock()
            return service

    @pytest.fixture
    def sample_coordinates(self):
        """Sample coordinates for testing"""
        return (-3.4653, -62.2159)  # Amazon Basin

    @pytest.fixture
    def sample_image_file(self):
        """Create temporary image file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(b'fake image data')
            yield tmp.name
        os.unlink(tmp.name)

    def test_initialization_success(self):
        """Test successful ML service initialization"""
        with patch('backend.services.ml_service.CarbonCreditVerificationPipeline') as mock_pipeline:
            mock_pipeline.return_value = Mock()
            service = MLService()
            assert service.is_initialized is True
            assert service.pipeline is not None

    def test_initialization_failure(self):
        """Test ML service initialization failure"""
        with patch('backend.services.ml_service.CarbonCreditVerificationPipeline', side_effect=Exception("Import error")):
            service = MLService()
            assert service.is_initialized is False
            assert service.pipeline is None

    @pytest.mark.asyncio
    async def test_analyze_location_success(self, ml_service, sample_coordinates):
        """Test successful location analysis"""
        project_id = 1
        result = await ml_service.analyze_location(sample_coordinates, project_id)
        
        assert result is not None
        assert result['project_id'] == project_id
        assert result['coordinates'] == list(sample_coordinates)
        assert 'forest_analysis' in result
        assert 'carbon_estimate' in result
        assert 'model_info' in result
        assert result['status'] == 'completed'

    @pytest.mark.asyncio
    async def test_analyze_location_not_initialized(self, sample_coordinates):
        """Test location analysis when service not initialized"""
        service = MLService()
        service.is_initialized = False
        
        with pytest.raises(MLServiceNotInitializedError, match="ML Service not initialized"):
            await service.analyze_location(sample_coordinates, 1)

    @pytest.mark.asyncio
    async def test_analyze_forest_cover_success(self, ml_service, sample_image_file):
        """Test successful forest cover analysis"""
        project_id = 1
        ml_service.pipeline.process_single_image.return_value = {
            'forest_coverage': 0.65,
            'confidence': 0.89,
            'processing_time': 15.2
        }
        
        result = await ml_service.analyze_forest_cover(sample_image_file, project_id)
        
        assert result is not None
        assert result['project_id'] == project_id
        assert result['analysis_type'] == 'forest_cover'
        ml_service.pipeline.process_single_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_forest_cover_failure(self, ml_service, sample_image_file):
        """Test forest cover analysis failure"""
        ml_service.pipeline.process_single_image.return_value = None
        
        with pytest.raises(MLAnalysisError, match="Forest cover analysis failed"):
            await ml_service.analyze_forest_cover(sample_image_file, 1)

    @pytest.mark.asyncio
    async def test_detect_changes_success(self, ml_service, sample_image_file):
        """Test successful change detection"""
        project_id = 1
        ml_service.pipeline.process_change_detection.return_value = {
            'change_percentage': 0.15,
            'confidence': 0.82,
            'processing_time': 25.5
        }
        
        result = await ml_service.detect_changes(sample_image_file, sample_image_file, project_id)
        
        assert result is not None
        assert result['project_id'] == project_id
        assert result['analysis_type'] == 'change_detection'
        ml_service.pipeline.process_change_detection.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_changes_failure(self, ml_service, sample_image_file):
        """Test change detection failure"""
        ml_service.pipeline.process_change_detection.return_value = None
        
        with pytest.raises(RuntimeError, match="Change detection analysis failed"):
            await ml_service.detect_changes(sample_image_file, sample_image_file, 1)

    @pytest.mark.asyncio
    async def test_analyze_time_series_success(self, ml_service):
        """Test successful time series analysis"""
        project_id = 1
        image_paths = ['path1.jpg', 'path2.jpg', 'path3.jpg']
        
        mock_result = {
            'trend': 'increasing',
            'confidence': 0.91,
            'processing_time': 45.3
        }
        ml_service.pipeline.process_temporal_sequence.return_value = mock_result
        
        result = await ml_service.analyze_time_series(image_paths, project_id)
        
        assert result is not None
        assert result['project_id'] == project_id
        assert result['analysis_type'] == 'time_series'
        ml_service.pipeline.process_temporal_sequence.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensemble_prediction_success(self, ml_service):
        """Test successful ensemble prediction"""
        project_id = 1
        image_data = {
            'single_image': 'path1.jpg',
            'before_image': 'path2.jpg',
            'after_image': 'path3.jpg',
            'time_series': ['path1.jpg', 'path2.jpg', 'path3.jpg']
        }
        
        ml_service.analyze_forest_cover = AsyncMock(return_value={'coverage': 0.65})
        ml_service.detect_changes = AsyncMock(return_value={'change': 0.15})
        ml_service.analyze_time_series = AsyncMock(return_value={'trend': 'stable'})
        
        result = await ml_service.ensemble_prediction(image_data, project_id)
        
        assert result is not None
        assert result['project_id'] == project_id
        assert result['analysis_type'] == 'ensemble'
        assert 'ensemble_confidence' in result
        assert 'final_recommendation' in result

    def test_calculate_ensemble_confidence(self, ml_service):
        """Test ensemble confidence calculation"""
        results = {
            'forest_analysis': {'confidence_score': 0.89},
            'change_detection': {'confidence': 0.82},
            'time_series': {'confidence': 0.91}
        }
        
        confidence = ml_service._calculate_ensemble_confidence(results)
        assert 0 <= confidence <= 1
        assert isinstance(confidence, float)

    def test_generate_recommendation(self, ml_service):
        """Test recommendation generation"""
        results = {
            'forest_analysis': {'forest_coverage_percent': 65.8},
            'carbon_estimate': {'total_carbon_tons': 2847.5},
            'ensemble_confidence': 0.87
        }
        
        recommendation = ml_service._generate_recommendation(results)
        
        assert 'carbon_credit_eligible' in recommendation
        assert 'confidence_level' in recommendation
        assert 'estimated_credits' in recommendation
        assert 'next_steps' in recommendation

    @pytest.mark.asyncio
    async def test_save_uploaded_file(self, ml_service):
        """Test file upload functionality"""
        file_content = b'fake file content'
        filename = 'test_image.jpg'
        
        file_path = await ml_service.save_uploaded_file(file_content, filename)
        
        assert file_path is not None
        assert filename in file_path
        assert os.path.exists(file_path)
        
        # Clean up
        os.unlink(file_path)

    def test_get_service_status(self, ml_service):
        """Test service status retrieval"""
        status = ml_service.get_service_status()
        
        assert 'initialized' in status
        assert 'models_loaded' in status
        assert 'pipeline_available' in status
        assert status['initialized'] is True

    def test_service_status_not_initialized(self):
        """Test service status when not initialized"""
        service = MLService()
        service.is_initialized = False
        service.pipeline = None  # Ensure pipeline is None
        
        status = service.get_service_status()
        
        assert status['initialized'] is False
        assert status['pipeline_available'] is False


class TestMLServiceIntegration:
    """Integration tests for ML Service with external dependencies"""

    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self):
        """Test complete analysis workflow"""
        with patch('backend.services.ml_service.CarbonCreditVerificationPipeline') as mock_pipeline:
            # Mock pipeline methods
            mock_instance = Mock()
            mock_instance.process_single_image.return_value = {'coverage': 0.65}
            mock_instance.process_change_detection.return_value = {'change': 0.15}
            mock_instance.process_time_series.return_value = {'trend': 'stable'}
            mock_pipeline.return_value = mock_instance
            
            service = MLService()
            
            # Test location analysis
            coords = (-3.4653, -62.2159)
            location_result = await service.analyze_location(coords, 1)
            assert location_result['status'] == 'completed'
            
            # Test forest cover analysis
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp.write(b'fake image')
                tmp.flush()
                
                forest_result = await service.analyze_forest_cover(tmp.name, 1)
                assert forest_result['analysis_type'] == 'forest_cover'
                
                os.unlink(tmp.name)

    def test_error_handling_robustness(self):
        """Test error handling across different failure scenarios"""
        # Test import failure
        with patch('backend.services.ml_service.CarbonCreditVerificationPipeline', side_effect=ImportError("No module")):
            service = MLService()
            assert service.is_initialized is False
        
        # Test runtime failure
        with patch('backend.services.ml_service.CarbonCreditVerificationPipeline', side_effect=RuntimeError("Runtime error")):
            service = MLService()
            assert service.is_initialized is False


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 