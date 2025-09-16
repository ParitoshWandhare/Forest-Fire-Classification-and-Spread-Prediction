#!/usr/bin/env python3
"""
Fixed training script for forest fire detection.

Key fixes:
1. Removed overly defensive config adaptation
2. Simplified data loader handling
3. Better error handling and logging
4. Proper tensor shape validation
5. Class imbalance handling
"""

import argparse
import sys
import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import warnings
import random
import numpy as np
from typing import Dict, Tuple, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import our modules
from models.unet import create_model
from datasets.fire_dataset import create_data_loaders, FireDataModule
from losses.dice_focal import create_loss_function
from train.trainer import create_trainer


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_optimizer(model: torch.nn.Module, config: dict) -> optim.Optimizer:
    """Create optimizer from configuration."""
    optimizer_type = config.get('type', 'adamw').lower()
    learning_rate = float(config.get('learning_rate', 1e-4))
    weight_decay = float(config.get('weight_decay', 1e-4))

    if optimizer_type == 'adamw':
        return optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay, 
            betas=config.get('betas', (0.9, 0.999))
        )
    elif optimizer_type == 'adam':
        return optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay, 
            betas=config.get('betas', (0.9, 0.999))
        )
    elif optimizer_type == 'sgd':
        return optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay, 
            momentum=config.get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def validate_batch_shapes(images: torch.Tensor, masks: torch.Tensor, batch_idx: int = 0):
    """Validate that batch has expected shapes and types."""
    print(f"\nBatch {batch_idx} validation:")
    print(f"  Images: {images.shape}, dtype: {images.dtype}, range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Masks: {masks.shape}, dtype: {masks.dtype}, range: [{masks.min():.3f}, {masks.max():.3f}]")
    
    # Check for expected shapes
    if images.dim() != 4:
        raise ValueError(f"Expected images to be 4D [B,C,H,W], got {images.shape}")
    if masks.dim() not in [3, 4]:
        raise ValueError(f"Expected masks to be 3D [B,H,W] or 4D [B,1,H,W], got {masks.shape}")
    
    # Check spatial dimensions match
    if images.shape[-2:] != masks.shape[-2:]:
        raise ValueError(f"Spatial dimensions don't match: images {images.shape[-2:]} vs masks {masks.shape[-2:]}")
    
    # Check for NaN or inf values
    if torch.isnan(images).any() or torch.isinf(images).any():
        raise ValueError(f"Images contain NaN or inf values")
    if torch.isnan(masks).any() or torch.isinf(masks).any():
        raise ValueError(f"Masks contain NaN or inf values")
    
    # Check mask values are in valid range
    if masks.min() < 0 or masks.max() > 1:
        print(f"  WARNING: Mask values outside [0,1] range: [{masks.min():.3f}, {masks.max():.3f}]")
        
    # Check class distribution
    positive_pixels = (masks > 0.5).sum().item()
    total_pixels = masks.numel()
    positive_ratio = positive_pixels / total_pixels
    print(f"  Fire pixels: {positive_pixels}/{total_pixels} ({positive_ratio*100:.2f}%)")
    
    if positive_ratio == 0:
        print(f"  WARNING: No positive pixels in batch {batch_idx}")
    elif positive_ratio > 0.5:
        print(f"  WARNING: Very high fire ratio ({positive_ratio*100:.1f}%) in batch {batch_idx}")


def print_model_info(model: torch.nn.Module, input_shape: tuple):
    """Print model architecture information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024**2:.1f} MB")
    
    # Test forward pass
    try:
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, *input_shape).to(device)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            if isinstance(output, dict):
                main_shape = output['out'].shape
                print(f"  Input shape: {dummy_input.shape}")
                print(f"  Output shape: {main_shape}")
                if 'aux' in output:
                    print(f"  Auxiliary outputs: {len(output['aux'])}")
            else:
                print(f"  Input shape: {dummy_input.shape}")
                print(f"  Output shape: {output.shape}")
        print("  Model validation: PASSED")
    except Exception as e:
        print(f"  Model validation: FAILED - {e}")


def main():
    parser = argparse.ArgumentParser(description='Train forest fire detection model')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to training configuration file')
    parser.add_argument('--data-config', type=str, required=True, 
                       help='Path to data configuration file')
    parser.add_argument('--model-config', type=str, required=True, 
                       help='Path to model configuration file')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', 
                       help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs', 
                       help='Log directory')
    parser.add_argument('--validate-data', action='store_true',
                       help='Run extensive data validation')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    print(f"Random seed: {args.seed}")
    
    # Load configurations
    try:
        train_config = load_config(args.config)
        data_config = load_config(args.data_config)
        model_config = load_config(args.model_config)
        print("Configurations loaded successfully")
    except Exception as e:
        print(f"Failed to load configurations: {e}")
        return 1
    
    # Create data configuration for data module
    data_module_config = {
        'data_root': data_config['data']['tiles_dir'],
        'batch_size': train_config.get('batch_size', 16),
        'image_size': data_config['data'].get('tile_size', 512),
        'val_split': data_config['data'].get('val_ratio', 0.2),
        'test_split': data_config['data'].get('test_ratio', 0.1),
        'num_workers': train_config.get('num_workers', 4),
        'use_weighted_sampling': True,
        'task': 'segmentation',
        'augmentation': train_config.get('augmentation', {})
    }
    
    # Create data loaders
    print("\nCreating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_config=data_module_config,
            train_config=train_config,
            num_workers=data_module_config['num_workers']
        )
        
        print(f"Data loaders created:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Validate first batch
        train_batch = next(iter(train_loader))
        images, masks = train_batch
        validate_batch_shapes(images, masks, 0)
        
        if args.validate_data:
            print("\nRunning extensive data validation...")
            for i, (imgs, msks) in enumerate(train_loader):
                validate_batch_shapes(imgs, msks, i)
                if i >= 2:  # Check first 3 batches
                    break
        
        input_shape = images.shape[1:]  # (C, H, W)
        
    except Exception as e:
        print(f"Failed to create data loaders: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create model
    print("\nCreating model...")
    try:
        model = create_model(model_config['model'])
        model = model.to(args.device)
        print_model_info(model, input_shape)
    except Exception as e:
        print(f"Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create loss function
    print("\nCreating loss function...")
    try:
        loss_fn = create_loss_function(train_config['loss'])
        print(f"Loss function: {train_config['loss'].get('loss_type', 'combined')}")
    except Exception as e:
        print(f"Failed to create loss function: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create optimizer
    print("\nCreating optimizer...")
    try:
        optimizer = create_optimizer(model, train_config['optimizer'])
        print(f"Optimizer: {train_config['optimizer'].get('type', 'adamw')}")
        print(f"Learning rate: {train_config['optimizer'].get('learning_rate', 1e-4)}")
    except Exception as e:
        print(f"Failed to create optimizer: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create trainer configuration
    trainer_config = {
        'epochs': train_config.get('epochs', 100),
        'mixed_precision': train_config.get('mixed_precision', False),
        'gradient_clipping': train_config.get('gradient_clipping', 1.0),
        'log_interval': train_config.get('log_interval', 10),
        'validation_interval': train_config.get('validation_interval', 1),
        'save_interval': train_config.get('save_interval', 5),
        'scheduler': train_config.get('scheduler', {}),
        'early_stopping': train_config.get('early_stopping', {}),
        'optimizer': train_config['optimizer'],
        'loss': train_config['loss']
    }
    
    # Create trainer
    print("\nSetting up trainer...")
    try:
        trainer = create_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            config=trainer_config,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir
        )
        print("Trainer setup complete")
    except Exception as e:
        print(f"Failed to setup trainer: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Start training
    print(f"\nStarting training for {trainer_config['epochs']} epochs...")
    print("=" * 60)
    
    try:
        training_results = trainer.train(resume_checkpoint=args.resume)
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        
        best_metric = training_results.get('best_val_metric', 0)
        total_epochs = training_results.get('total_epochs', 0)
        total_time = training_results.get('total_time', 0)
        
        print(f"Best validation metric: {best_metric:.4f}")
        print(f"Total epochs: {total_epochs}")
        print(f"Total time: {total_time/3600:.2f} hours")
        
        # Save training summary
        summary = {
            'results': training_results,
            'config': {
                'train': train_config,
                'data': data_config,
                'model': model_config
            },
            'args': vars(args)
        }
        
        summary_path = Path(args.log_dir) / 'training_summary.yaml'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, indent=2)
        
        print(f"Training summary saved to: {summary_path}")
        return 0
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)