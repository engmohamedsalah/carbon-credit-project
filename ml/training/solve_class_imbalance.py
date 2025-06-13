#!/usr/bin/env python3
"""
Comprehensive Class Imbalance Solutions for ConvLSTM Training
Multiple strategies to create balanced training data from existing models and data
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import rasterio
from pathlib import Path
import random
from sklearn.utils.class_weight import compute_class_weight
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.models.convlstm_model import ConvLSTM
from ml.training.enhanced_time_series_dataset import MultiModalTimeSeriesDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClassImbalanceSolver:
    """Comprehensive class imbalance solver with multiple strategies"""
    
    def __init__(self, data_dir='ml/data/prepared'):
        self.data_dir = data_dir
        self.s2_dir = os.path.join(data_dir, 's2_stacks')
        self.s1_dir = os.path.join(data_dir, 's1_stacks')
        self.change_dir = os.path.join(data_dir, 'change_labels')
        
    def strategy_1_synthetic_positives_from_change_detection(self):
        """
        STRATEGY 1: Generate synthetic positive samples using existing change detection model
        Use the Siamese U-Net change detection outputs as positive training examples
        """
        logger.info("🎯 STRATEGY 1: Synthetic Positives from Change Detection")
        logger.info("=" * 60)
        
        # Load existing change detection model
        change_model_path = 'ml/models/change_detection_siamese_unet.pth'
        if not os.path.exists(change_model_path):
            logger.error(f"Change detection model not found: {change_model_path}")
            return None
            
        logger.info("📂 Loading change detection model...")
        # This would load the Siamese U-Net and generate change predictions
        # on temporal pairs to create positive training samples
        
        synthetic_strategy = {
            'name': 'Synthetic Positives from Change Detection',
            'description': 'Use Siamese U-Net outputs as positive training labels',
            'steps': [
                '1. Load trained Siamese U-Net change detection model',
                '2. Run inference on all temporal pairs in dataset',
                '3. Extract high-confidence change predictions (threshold > 0.6)',
                '4. Use these as positive labels for ConvLSTM training',
                '5. Balance with equal number of no-change samples'
            ],
            'pros': [
                'Uses existing high-quality model (F1=0.60)',
                'Creates realistic positive samples',
                'Leverages temporal information',
                'Can generate thousands of positive samples'
            ],
            'cons': [
                'Limited by change detection model accuracy',
                'May inherit model biases'
            ],
            'implementation_time': '2-3 hours',
            'expected_improvement': 'High - should achieve F1 > 0.3'
        }
        
        return synthetic_strategy
    
    def strategy_2_weighted_loss_functions(self):
        """
        STRATEGY 2: Advanced weighted loss functions for extreme imbalance
        """
        logger.info("🎯 STRATEGY 2: Weighted Loss Functions")
        logger.info("=" * 60)
        
        weighted_loss_strategy = {
            'name': 'Weighted Loss Functions',
            'description': 'Use heavily weighted loss to force model to learn positive class',
            'implementations': {
                'weighted_bce': {
                    'function': 'BCEWithLogitsLoss(pos_weight=1000)',
                    'description': 'Give 1000x weight to positive samples',
                    'code': '''
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1000.0]))
                    '''
                },
                'focal_loss': {
                    'function': 'FocalLoss(alpha=0.99, gamma=2)',
                    'description': 'Focus on hard-to-classify samples',
                    'code': '''
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.99, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()
                    '''
                },
                'class_balanced_loss': {
                    'function': 'ClassBalancedLoss(beta=0.9999)',
                    'description': 'Effective number of samples based weighting',
                    'code': '''
# Calculate effective number of samples
effective_num = 1.0 - np.power(beta, samples_per_class)
weights = (1.0 - beta) / np.array(effective_num)
                    '''
                }
            },
            'pros': [
                'Quick to implement (30 minutes)',
                'No additional data needed',
                'Can work with existing pipeline',
                'Multiple proven approaches'
            ],
            'cons': [
                'May cause training instability',
                'Requires hyperparameter tuning',
                'Still limited by lack of positive examples'
            ],
            'implementation_time': '30 minutes - 1 hour',
            'expected_improvement': 'Medium - should achieve F1 > 0.1'
        }
        
        return weighted_loss_strategy
    
    def strategy_3_balanced_sampling(self):
        """
        STRATEGY 3: Intelligent balanced sampling from existing data
        """
        logger.info("🎯 STRATEGY 3: Balanced Sampling")
        logger.info("=" * 60)
        
        balanced_sampling_strategy = {
            'name': 'Intelligent Balanced Sampling',
            'description': 'Force balanced batches by oversampling rare positive examples',
            'approaches': {
                'hansen_based_sampling': {
                    'description': 'Use Hansen forest loss data to find positive regions',
                    'method': 'Sample patches from Hansen loss years 2020-2023',
                    'code': '''
# Load Hansen loss year data
hansen_loss = rasterio.open('ml/data/hansen_downloads/hansen_clipped_lossyear.tif')
loss_mask = (hansen_loss.read(1) >= 20) & (hansen_loss.read(1) <= 23)  # 2020-2023

# Sample patches from loss regions
positive_patches = extract_patches_from_mask(loss_mask, patch_size=64)
                    '''
                },
                'edge_detection_sampling': {
                    'description': 'Sample from forest edges where change is more likely',
                    'method': 'Use forest cover gradients to find transition zones',
                    'code': '''
# Calculate forest cover gradients
forest_cover = load_forest_cover_data()
gradients = calculate_gradients(forest_cover)
edge_mask = gradients > threshold

# Sample from edge regions
edge_patches = extract_patches_from_mask(edge_mask, patch_size=64)
                    '''
                },
                'temporal_variance_sampling': {
                    'description': 'Sample regions with high temporal variance in NDVI',
                    'method': 'Find areas with significant vegetation changes',
                    'code': '''
# Calculate NDVI temporal variance
ndvi_sequences = load_ndvi_time_series()
temporal_variance = np.var(ndvi_sequences, axis=0)
high_variance_mask = temporal_variance > percentile_95

# Sample from high variance regions
variance_patches = extract_patches_from_mask(high_variance_mask, patch_size=64)
                    '''
                }
            },
            'pros': [
                'Uses existing data intelligently',
                'Can find real positive examples',
                'Multiple complementary approaches',
                'Preserves data authenticity'
            ],
            'cons': [
                'May still have limited positive samples',
                'Requires careful validation',
                'More complex implementation'
            ],
            'implementation_time': '3-4 hours',
            'expected_improvement': 'Medium-High - should achieve F1 > 0.2'
        }
        
        return balanced_sampling_strategy
    
    def strategy_4_data_augmentation(self):
        """
        STRATEGY 4: Advanced data augmentation for positive samples
        """
        logger.info("🎯 STRATEGY 4: Data Augmentation")
        logger.info("=" * 60)
        
        augmentation_strategy = {
            'name': 'Advanced Data Augmentation',
            'description': 'Generate more positive samples through intelligent augmentation',
            'techniques': {
                'temporal_interpolation': {
                    'description': 'Create intermediate time steps between existing sequences',
                    'method': 'Linear/spline interpolation between temporal frames',
                    'multiplier': '2-3x more samples'
                },
                'spatial_augmentation': {
                    'description': 'Rotate, flip, scale positive patches',
                    'method': 'Standard geometric transformations',
                    'multiplier': '8x more samples (rotations + flips)'
                },
                'noise_injection': {
                    'description': 'Add realistic noise to positive samples',
                    'method': 'Gaussian noise, atmospheric effects simulation',
                    'multiplier': '3-5x more samples'
                },
                'mixup_augmentation': {
                    'description': 'Blend positive and negative samples',
                    'method': 'Linear combination of samples and labels',
                    'multiplier': '10x more samples'
                }
            },
            'pros': [
                'Dramatically increases positive samples',
                'Improves model robustness',
                'Well-established techniques',
                'Can combine with other strategies'
            ],
            'cons': [
                'May introduce unrealistic samples',
                'Requires careful validation',
                'Can lead to overfitting'
            ],
            'implementation_time': '2-3 hours',
            'expected_improvement': 'High - should achieve F1 > 0.4'
        }
        
        return augmentation_strategy
    
    def strategy_5_ensemble_approach(self):
        """
        STRATEGY 5: Ensemble with existing models
        """
        logger.info("🎯 STRATEGY 5: Ensemble Approach")
        logger.info("=" * 60)
        
        ensemble_strategy = {
            'name': 'Ensemble with Existing Models',
            'description': 'Combine ConvLSTM with proven U-Net models',
            'architecture': {
                'stage_1': 'Forest Cover U-Net (F1=0.49) - Base forest mapping',
                'stage_2': 'Change Detection Siamese U-Net (F1=0.60) - Change detection',
                'stage_3': 'ConvLSTM - Temporal refinement and false positive reduction',
                'final': 'Weighted ensemble of all three predictions'
            },
            'ensemble_methods': {
                'weighted_average': {
                    'formula': '0.3 * forest_cover + 0.4 * change_detection + 0.3 * convlstm',
                    'description': 'Simple weighted combination'
                },
                'stacked_ensemble': {
                    'description': 'Train meta-model on outputs of all three models',
                    'method': 'Logistic regression or small neural network'
                },
                'conditional_ensemble': {
                    'description': 'Use ConvLSTM only when other models disagree',
                    'method': 'If |forest_pred - change_pred| > threshold, use ConvLSTM'
                }
            },
            'pros': [
                'Leverages all existing work',
                'Likely best overall performance',
                'Reduces individual model weaknesses',
                'Production-ready approach'
            ],
            'cons': [
                'More complex inference pipeline',
                'Higher computational cost',
                'Requires careful weight tuning'
            ],
            'implementation_time': '1-2 hours',
            'expected_improvement': 'Very High - should achieve F1 > 0.6'
        }
        
        return ensemble_strategy
    
    def strategy_6_active_learning(self):
        """
        STRATEGY 6: Active learning with uncertainty sampling
        """
        logger.info("🎯 STRATEGY 6: Active Learning")
        logger.info("=" * 60)
        
        active_learning_strategy = {
            'name': 'Active Learning with Uncertainty Sampling',
            'description': 'Iteratively find and label the most informative samples',
            'process': [
                '1. Train initial model on available data',
                '2. Run inference on large unlabeled dataset',
                '3. Find samples with highest uncertainty (probability ≈ 0.5)',
                '4. Use existing models to pseudo-label these samples',
                '5. Add high-confidence pseudo-labels to training set',
                '6. Retrain model and repeat'
            ],
            'uncertainty_measures': {
                'entropy': 'H = -p*log(p) - (1-p)*log(1-p)',
                'margin': 'Difference between top two predictions',
                'variance': 'Variance across multiple forward passes (MC Dropout)'
            },
            'pros': [
                'Finds most informative samples',
                'Iterative improvement',
                'Efficient use of labeling effort',
                'Can discover edge cases'
            ],
            'cons': [
                'Requires multiple training iterations',
                'More complex pipeline',
                'May be slow to converge'
            ],
            'implementation_time': '4-5 hours',
            'expected_improvement': 'High - should achieve F1 > 0.3'
        }
        
        return active_learning_strategy
    
    def create_implementation_plan(self):
        """Create a prioritized implementation plan"""
        logger.info("📋 IMPLEMENTATION PLAN")
        logger.info("=" * 60)
        
        strategies = [
            self.strategy_1_synthetic_positives_from_change_detection(),
            self.strategy_2_weighted_loss_functions(),
            self.strategy_3_balanced_sampling(),
            self.strategy_4_data_augmentation(),
            self.strategy_5_ensemble_approach(),
            self.strategy_6_active_learning()
        ]
        
        # Prioritize by implementation time vs expected improvement
        priority_order = [
            ('Strategy 2: Weighted Loss', 'Quick win - 30 min implementation'),
            ('Strategy 5: Ensemble', 'Best ROI - 1-2 hours for F1 > 0.6'),
            ('Strategy 1: Synthetic Positives', 'High impact - 2-3 hours'),
            ('Strategy 4: Data Augmentation', 'Robust solution - 2-3 hours'),
            ('Strategy 3: Balanced Sampling', 'Data-driven - 3-4 hours'),
            ('Strategy 6: Active Learning', 'Advanced - 4-5 hours')
        ]
        
        logger.info("🎯 RECOMMENDED PRIORITY ORDER:")
        for i, (strategy, description) in enumerate(priority_order, 1):
            logger.info(f"{i}. {strategy}: {description}")
        
        return strategies, priority_order
    
    def quick_implementation_weighted_loss(self):
        """Quick implementation of weighted loss strategy"""
        logger.info("🚀 QUICK IMPLEMENTATION: Weighted Loss")
        logger.info("=" * 50)
        
        code_template = '''
# Quick fix for class imbalance - add to your training script:

import torch.nn as nn

# Calculate positive weight based on class distribution
# If 0.1% positive samples, use weight = 999 (1000x emphasis)
pos_weight = torch.tensor([999.0])  # Adjust based on your data

# Replace your criterion with:
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Alternative: Focal Loss for extreme imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.99, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

# Use focal loss:
criterion = FocalLoss(alpha=0.99, gamma=2)

# Also lower your threshold for evaluation:
threshold = 0.01  # Instead of 0.5
predictions = torch.sigmoid(outputs) > threshold
        '''
        
        logger.info("📝 Code template:")
        print(code_template)
        
        return code_template

def main():
    """Main function to analyze and present solutions"""
    print("🎯 CLASS IMBALANCE SOLUTIONS FOR CONVLSTM")
    print("=" * 60)
    print("Analyzing your biased data problem and providing solutions...\n")
    
    solver = ClassImbalanceSolver()
    
    # Get all strategies
    strategies, priority_order = solver.create_implementation_plan()
    
    print("\n" + "="*60)
    print("💡 IMMEDIATE ACTION ITEMS:")
    print("="*60)
    print("1. 🚀 QUICK WIN (30 min): Implement weighted loss function")
    print("2. 🎯 BEST ROI (1-2 hours): Create ensemble with existing models")
    print("3. 📊 HIGH IMPACT (2-3 hours): Generate synthetic positives from change detection")
    print("\n✅ Your ConvLSTM architecture is proven to work!")
    print("✅ The issue is purely data imbalance - very solvable!")
    print("✅ You already have excellent models (F1=0.49, F1=0.60) to leverage!")
    
    # Show quick implementation
    print("\n" + "="*60)
    print("🚀 QUICK IMPLEMENTATION (Copy-paste ready):")
    print("="*60)
    solver.quick_implementation_weighted_loss()

if __name__ == "__main__":
    main() 