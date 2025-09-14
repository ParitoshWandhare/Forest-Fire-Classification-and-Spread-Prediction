#!/usr/bin/env python3
"""
Training script for forest fire detection using U-Net.

This script has been updated to match the hardened dataset, loss and trainer
APIs. Key improvements:
- Rigorous config adaptation and type conversion
- Sanity checks on data loader batches (shapes / dtypes)
- Ensure mask dtype matches loss expectations (float for BCE/BCEWithLogits, long for cross_entropy)
- Safer device selection and checkpoint handling
- Clearer logging and error messages

Usage example remains similar to before.
"""

import argparse
import sys
import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from pathlib import Path
import warnings
import random
import numpy as np
from typing import Dict, Tuple, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import our modules
from src.models.unet import create_model
from src.datasets.fire_dataset import create_data_loaders, FireDataModule
from src.losses.dice_focal import create_loss_function
from src.train.trainer import create_trainer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def adapt_data_config(config):
    data_section = config.get('data', {})

    adapted = {
        'data_root': data_section.get('tiles_dir', 'data/interim/tiles'),
        'image_size': data_section.get('tile_size', 512),
        'batch_size': 16,
        'task': 'segmentation',
        'val_split': data_section.get('val_ratio', 0.2),
        'test_split': data_section.get('test_ratio', 0.1),
        'use_weighted_sampling': True,
        'num_workers': 4,
        'in_channels': 3,
        'augmentation': {
            'enabled': True,
            'horizontal_flip': 0.5,
            'vertical_flip': 0.5,
            'rotation_90': 0.5,
            'brightness_contrast': 0.3,
            'gaussian_noise': 0.2,
            'cutout': 0.2,
            'brightness_limit': 0.2,
            'contrast_limit': 0.2,
            'noise_var_limit': [0.0, 0.02],
            'cutout_holes': 8,
            'cutout_size': 32
        }
    }
    return adapted


def adapt_model_config(config):
    model_section = config.get('model', {})
    unet_section = model_section.get('unet', {})

    adapted = {
        'architecture': 'unet',
        'backbone': unet_section.get('encoder_name', 'resnet34'),
        'pretrained': unet_section.get('encoder_weights') == 'imagenet',
        'in_channels': unet_section.get('in_channels', 3),
        'num_classes': unet_section.get('classes', 1),
        'decoder_channels': unet_section.get('decoder_channels', [256, 128, 64]),
        'dropout': 0.1,
        'aux_outputs': True
    }
    return adapted


def adapt_train_config(config):
    training_section = config.get('training', {})

    opt_config = training_section.get('optimizer', {})
    lr_value = opt_config.get('lr', 1e-4)
    if isinstance(lr_value, str):
        lr_value = float(lr_value)

    wd_value = opt_config.get('weight_decay', 1e-4)
    if isinstance(wd_value, str):
        wd_value = float(wd_value)

    optimizer_adapted = {
        'type': opt_config.get('name', 'adam'),
        'learning_rate': float(lr_value),
        'weight_decay': float(wd_value),
        'betas': opt_config.get('betas', [0.9, 0.999])
    }

    sched_config = training_section.get('scheduler', {})
    scheduler_adapted = {
        'type': sched_config.get('name', 'cosine'),
        'T_max': int(sched_config.get('T_max', 100)),
        'eta_min': float(sched_config.get('eta_min', 1e-6)),
        'step_size': int(sched_config.get('step_size', 30)),
        'gamma': float(sched_config.get('gamma', 0.1)),
        'patience': int(sched_config.get('patience', 10)),
        'factor': float(sched_config.get('factor', 0.5))
    }

    loss_config = training_section.get('loss', {})
    loss_adapted = {
        'loss_type': loss_config.get('name', 'combined'),
        'dice_weight': float(loss_config.get('dice_weight', 0.5)),
        'focal_weight': float(loss_config.get('focal_weight', 0.5)),
        'focal_alpha': float(loss_config.get('focal_alpha', 0.25)),
        'focal_gamma': float(loss_config.get('focal_gamma', 2.0)),
        'aux_weight': float(loss_config.get('aux_weight', 0.3))
    }

    es_config = training_section.get('early_stopping', {})
    early_stopping_adapted = {
        'patience': int(es_config.get('patience', 15)),
        'min_delta': float(es_config.get('min_delta', 0.001)),
        'mode': es_config.get('mode', 'max'),
        'restore_best_weights': True
    }

    adapted = {
        'epochs': int(training_section.get('epochs', 100)),
        'batch_size': int(training_section.get('batch_size', 16)),
        'num_workers': int(training_section.get('num_workers', 4)),
        'mixed_precision': training_section.get('precision', 32) == 16,
        'gradient_clipping': float(training_section.get('gradient_clip_val', 1.0)),
        'log_interval': int(training_section.get('logging', {}).get('log_every_n_steps', 10)),
        'validation_interval': int(training_section.get('validation', {}).get('frequency', 1)),
        'save_interval': int(training_section.get('checkpoint', {}).get('save_top_k', 5)),
        'optimizer': optimizer_adapted,
        'scheduler': scheduler_adapted,
        'loss': loss_adapted,
        'early_stopping': early_stopping_adapted
    }
    return adapted


def create_optimizer(model: torch.nn.Module, config: dict) -> optim.Optimizer:
    optimizer_config = config.get('optimizer', {})
    optimizer_type = optimizer_config.get('type', 'adamw').lower()
    learning_rate = optimizer_config.get('learning_rate', 1e-3)
    weight_decay = optimizer_config.get('weight_decay', 1e-4)

    if optimizer_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=optimizer_config.get('betas', (0.9, 0.999)))
    elif optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=optimizer_config.get('betas', (0.9, 0.999)))
    elif optimizer_type == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=optimizer_config.get('momentum', 0.9))
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def print_model_info(model: torch.nn.Module, input_shape: tuple):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Architecture Information:")
    print(f"{'='*50}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (float32)")

    try:
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, *input_shape).to(device)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            if isinstance(output, dict):
                main_output_shape = output['out'].shape
                print(f"Input shape: (1, {', '.join(map(str, input_shape))})")
                print(f"Output shape: {main_output_shape}")
            else:
                print(f"Input shape: (1, {', '.join(map(str, input_shape))})")
                print(f"Output shape: {output.shape}")
        print("✅ Model architecture validated")
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
    print(f"{'='*50}")


def print_training_summary(train_config: dict, data_config: dict, model_config: dict):
    print(f"\nTraining Configuration Summary:")
    print(f"{'='*60}")
    print(f"Dataset:")
    print(f"  - Data root: {data_config.get('data_root', 'N/A')}")
    print(f"  - Batch size: {data_config.get('batch_size', train_config.get('batch_size', 'N/A'))}")
    print(f"  - Image size: {data_config.get('image_size', 'N/A')}")
    print(f"  - Augmentation: {data_config.get('augmentation', {}).get('enabled', False)}")
    print(f"\nModel:")
    print(f"  - Architecture: {model_config.get('architecture', 'N/A')}")
    print(f"  - Backbone: {model_config.get('backbone', 'N/A')}")
    print(f"  - Pretrained: {model_config.get('pretrained', 'N/A')}")
    print(f"\nTraining:")
    print(f"  - Epochs: {train_config.get('epochs', 'N/A')}")
    print(f"  - Optimizer: {train_config.get('optimizer', {}).get('type', 'N/A')}")
    print(f"  - Learning rate: {train_config.get('optimizer', {}).get('learning_rate', 'N/A')}")
    print(f"{'='*60}")


def validate_config_compatibility(train_config: dict, data_config: dict, model_config: dict) -> bool:
    errors = []
    if 'training' not in train_config:
        errors.append("Train config must have 'training' section")
    if 'model' not in model_config:
        errors.append("Model config must have 'model' section")
    if 'data' not in data_config:
        errors.append("Data config must have 'data' section")
    if errors:
        print("❌ Configuration validation failed:")
        for error in errors:
            print(f"   - {error}")
        return False
    print("✅ Configuration validation passed")
    return True


# -------------------------
# DataLoader compatibility wrapper
# -------------------------
def _make_mask_compatible_collate(squeeze_channel: bool = True, target_dtype=torch.float32):
    """
    Returns a collate_fn that will:
      - call default_collate
      - ensure masks are shaped [B, H, W] (squeeze channel dim if present)
      - cast mask dtype to requested dtype
    """
    def collate_fn(batch):
        batch_collated = default_collate(batch)
        # expect (images, masks) or a tuple-like result
        if isinstance(batch_collated, (list, tuple)) and len(batch_collated) >= 2:
            images, masks = batch_collated[0], batch_collated[1]
            # if masks have shape [B,1,H,W], squeeze -> [B,H,W]
            if squeeze_channel and masks.dim() == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)
            # cast dtype if necessary
            if masks.dtype != target_dtype:
                try:
                    masks = masks.to(dtype=target_dtype)
                except Exception:
                    masks = masks.float()
            # rebuild the tuple
            return images, masks
        return batch_collated
    return collate_fn


def make_compatible_loader(orig_loader: DataLoader, squeeze_channel: bool = True,
                           target_dtype=torch.float32) -> DataLoader:
    """
    Create a new DataLoader with same dataset and settings but a collate_fn
    that fixes mask shape and dtype. Keeps sampler/batch_sampler if present.
    """
    ds = orig_loader.dataset
    params = {
        'dataset': ds,
        'batch_size': getattr(orig_loader, 'batch_size', None),
        'num_workers': getattr(orig_loader, 'num_workers', 0),
        'pin_memory': getattr(orig_loader, 'pin_memory', False),
        'drop_last': getattr(orig_loader, 'drop_last', False),
        'collate_fn': _make_mask_compatible_collate(squeeze_channel=squeeze_channel, target_dtype=target_dtype)
    }

    # preserve sampler / shuffle preference when possible
    if getattr(orig_loader, 'batch_sampler', None) is not None:
        # If there is a batch_sampler, use it (cannot set batch_size simultaneously)
        return DataLoader(dataset=ds, batch_sampler=orig_loader.batch_sampler,
                          num_workers=params['num_workers'], pin_memory=params['pin_memory'],
                          collate_fn=params['collate_fn'])
    else:
        # Use same sampler if present; otherwise set shuffle if available
        sampler = getattr(orig_loader, 'sampler', None)
        shuffle = getattr(orig_loader, 'shuffle', False)
        return DataLoader(dataset=ds, batch_size=params['batch_size'], shuffle=shuffle, sampler=sampler,
                          num_workers=params['num_workers'], pin_memory=params['pin_memory'],
                          drop_last=params['drop_last'], collate_fn=params['collate_fn'])


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description='Train forest fire detection model')
    parser.add_argument('--config', type=str, required=True, help='Path to training configuration file')
    parser.add_argument('--data-config', type=str, required=True, help='Path to data configuration file')
    parser.add_argument('--model-config', type=str, required=True, help='Path to model configuration file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (small dataset)')
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"🌱 Set random seed to {args.seed}")

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print(f"🖥️  Using device: {args.device}")

    print("\n📋 Loading configurations...")
    try:
        train_config_raw = load_config(args.config)
        data_config_raw = load_config(args.data_config)
        model_config_raw = load_config(args.model_config)
        print("✅ Configurations loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load configurations: {e}")
        return 1

    # --- Normalise configs: support both wrapped (training:, model:, data:) and flat formats ---
    def _normalize_configs(train_cfg, data_cfg, model_cfg):
        # Defensive defaults
        train_cfg = train_cfg or {}
        data_cfg = data_cfg or {}
        model_cfg = model_cfg or {}

        # If top-level 'training' missing but keys look like training keys, wrap them
        training_keys = {'epochs', 'batch_size', 'optimizer', 'loss', 'early_stopping', 'validation'}
        if 'training' not in train_cfg and any(k in train_cfg for k in training_keys):
            train_cfg = {'training': train_cfg}

        # If model top-level missing, try to detect model-like keys and wrap
        model_keys = {'unet', 'encoder_name', 'input_size', 'name', 'architecture', 'classes'}
        if 'model' not in model_cfg and any(k in model_cfg for k in model_keys):
            model_cfg = {'model': model_cfg}

        # If data top-level missing, try to detect data-like keys and wrap
        data_keys = {'tiles_dir', 'tile_size', 'val_ratio', 'train_dir', 'data_root', 'metadata_csv'}
        if 'data' not in data_cfg and any(k in data_cfg for k in data_keys):
            data_cfg = {'data': data_cfg}

        return train_cfg, data_cfg, model_cfg

    train_config_raw, data_config_raw, model_config_raw = _normalize_configs(
        train_config_raw, data_config_raw, model_config_raw
    )

    # Debugging: show what top-level keys we have now
    try:
        print("Config summary after normalization:")
        print(f"  - training present: {'training' in train_config_raw}")
        print(f"  - data present: {'data' in data_config_raw}")
        print(f"  - model present: {'model' in model_config_raw}")
    except Exception:
        pass

    # Validate after normalization
    if not validate_config_compatibility(train_config_raw, data_config_raw, model_config_raw):
        return 1


    train_config = adapt_train_config(train_config_raw)
    data_config = adapt_data_config(data_config_raw)
    model_config = adapt_model_config(model_config_raw)

    print("✅ Configurations adapted successfully")

    if args.num_workers is not None:
        train_config['num_workers'] = args.num_workers
        data_config['num_workers'] = args.num_workers

    print("\n📊 Creating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_config=data_config,
            train_config=train_config,
            num_workers=args.num_workers
        )

        print(f"✅ Data loaders created: Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

        sample_batch = next(iter(train_loader))
        if isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 2:
            sample_images, sample_masks = sample_batch[0], sample_batch[1]
        else:
            raise ValueError("Unexpected batch format from train loader")

        input_shape = sample_images.shape[1:]
        print(f"   Batch size: {sample_images.shape[0]}")
        print(f"   Input shape: {input_shape}")
        print(f"   Target shape: {sample_masks.shape}")
        print(f"   Image dtype: {sample_images.dtype}, Mask dtype: {sample_masks.dtype}")

        # Sanity: ensure mask shape/dtype will be compatible with loss & metrics
        # If masks come as [B,1,H,W] and loss expects [B,H,W] for BCEWithLogits, convert loader.
        # We'll assume BCE-like (binary) targets should be float32 and squeezed to [B,H,W].
        masks_need_squeeze = (sample_masks.dim() == 4 and sample_masks.shape[1] == 1)
        masks_not_float = (sample_masks.dtype not in (torch.float32, torch.float64))
        if masks_need_squeeze or masks_not_float:
            print("ℹ️  Converting data loader output masks to shape [B,H,W] and dtype float32 for loss compatibility.")
            train_loader = make_compatible_loader(train_loader, squeeze_channel=True, target_dtype=torch.float32)
            val_loader = make_compatible_loader(val_loader, squeeze_channel=True, target_dtype=torch.float32)
            test_loader = make_compatible_loader(test_loader, squeeze_channel=True, target_dtype=torch.float32)

            # Re-sample a batch to confirm
            sample_batch = next(iter(train_loader))
            sample_images, sample_masks = sample_batch[0], sample_batch[1]
            print(f"   (post-adapt) Target shape: {sample_masks.shape}, dtype: {sample_masks.dtype}")

    except Exception as e:
        print(f"❌ Failed to create data loaders: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n🏗️  Creating model...")
    try:
        model = create_model(model_config)
        model = model.to(args.device)
        print(f"✅ Model created: {model_config.get('architecture', 'Unknown')} with {model_config.get('backbone', 'Unknown')} backbone")
        print_model_info(model, input_shape)
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n🎯 Creating loss function...")
    try:
        loss_config = train_config.get('loss', {})
        loss_fn = create_loss_function(loss_config)
        print(f"✅ Loss function created: {loss_config.get('loss_type', 'combined')}")
    except Exception as e:
        print(f"❌ Failed to create loss function: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n⚙️  Creating optimizer...")
    try:
        optimizer = create_optimizer(model, train_config)
        print("✅ Optimizer created")
    except Exception as e:
        print(f"❌ Failed to create optimizer: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n🚀 Setting up trainer...")
    try:
        trainer = create_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            config=train_config,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir
        )
        print("✅ Trainer setup complete")
    except Exception as e:
        print(f"❌ Failed to setup trainer: {e}")
        import traceback
        traceback.print_exc()
        return 1

    if args.debug:
        print("\n🐛 Debug mode enabled - limiting dataset size and epochs")
        train_config['epochs'] = min(train_config.get('epochs', 100), 3)

    print(f"\n🔥 Starting training...")
    print(f"{'='*80}")

    try:
        training_results = trainer.train(resume_checkpoint=args.resume)

        print(f"\n{'='*80}")
        print("🎉 Training completed successfully!")
        best_metric = training_results.get('best_val_metric', 0)
        total_epochs = training_results.get('total_epochs', 0)
        total_time = training_results.get('total_time', 0)

        print(f"📊 Training Summary:")
        print(f"   Best validation metric: {best_metric:.4f}")
        print(f"   Total epochs: {total_epochs}")
        print(f"   Total time: {total_time/3600:.2f} hours")

        summary = {'config': {'train': train_config, 'data': data_config, 'model': model_config}, 'results': training_results, 'args': vars(args)}
        summary_path = Path(args.log_dir) / 'training_summary.yaml'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, indent=2, default_flow_style=False)

        print(f"💾 Training summary saved to: {summary_path}")
        return 0

    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)











# #!/usr/bin/env python3
# """
# Training script for forest fire detection using U-Net.

# This script provides a complete training pipeline:
# - Load configuration files
# - Initialize model, data loaders, loss function, optimizer
# - Set up training infrastructure
# - Run training with comprehensive monitoring
# - Save results and generate training reports

# Usage:
#     python scripts/train.py --config configs/train.yaml --data-config configs/data.yaml --model-config configs/model.yaml

# Example:
#     python scripts/trains.py \
#         --config configs/train.yaml \
#         --data-config configs/data.yaml \
#         --model-config configs/model.yaml \
#         --resume checkpoints/latest_checkpoint.pth \
#         --device cuda \
#         --num-workers 4
# """

# import argparse
# import sys
# import os
# import yaml
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from pathlib import Path
# import warnings
# import random
# import numpy as np
# from typing import Dict, Tuple, Optional

# # Add src to path
# sys.path.append(str(Path(__file__).parent.parent / 'src'))

# # Import our modules
# from models.unet import create_model
# from datasets.fire_dataset import create_data_loaders, FireDataModule
# from losses.dice_focal import create_loss_function
# from train.trainer import create_trainer


# def set_seed(seed: int = 42):
#     """Set seeds for reproducibility."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# def load_config(config_path: str) -> dict:
#     """Load YAML configuration file."""
#     config_path = Path(config_path)
#     if not config_path.exists():
#         raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
#     with open(config_path, 'r') as f:
#         return yaml.safe_load(f)


# def adapt_data_config(config):
#     """Adapt data config from nested structure to flat structure expected by code."""
#     data_section = config.get('data', {})
    
#     # Create flat structure expected by FireDataModule
#     adapted = {
#         'data_root': data_section.get('tiles_dir', 'data/interim/tiles'),
#         'image_size': data_section.get('tile_size', 512),
#         'batch_size': 16,  # Default
#         'task': 'segmentation',
#         'val_split': 0.2,
#         'test_split': 0.1,
#         'use_weighted_sampling': True,
#         'num_workers': 4,
#         'in_channels': 3,
#         'augmentation': {
#             'enabled': data_section.get('augmentation', {}).get('horizontal_flip', False),
#             'horizontal_flip': 0.5,
#             'vertical_flip': 0.5,
#             'rotation_90': 0.5,
#             'brightness_contrast': 0.3,
#             'gaussian_noise': 0.2,
#             'cutout': 0.2,
#             'brightness_limit': 0.2,
#             'contrast_limit': 0.2,
#             'noise_var_limit': [0.0, 0.05],
#             'cutout_holes': 8,
#             'cutout_size': 32
#         }
#     }
#     return adapted


# def adapt_model_config(config):
#     """Adapt model config from nested structure to flat structure expected by code."""
#     model_section = config.get('model', {})
#     unet_section = model_section.get('unet', {})
    
#     # Map from YAML structure to expected structure
#     adapted = {
#         'architecture': 'unet',
#         'backbone': unet_section.get('encoder_name', 'resnet34'),
#         'pretrained': unet_section.get('encoder_weights') == 'imagenet',
#         'in_channels': unet_section.get('in_channels', 3),
#         'num_classes': unet_section.get('classes', 1),
#         'decoder_channels': unet_section.get('decoder_channels', [256, 128, 64]),
#         'dropout': 0.1,
#         'aux_outputs': True
#     }
#     return adapted


# def adapt_train_config(config):
#     """Adapt training config from nested structure to flat structure expected by code."""
#     training_section = config.get('training', {})
    
#     # Extract optimizer config with proper type conversion
#     opt_config = training_section.get('optimizer', {})
    
#     # Handle learning rate - ensure it's a float
#     lr_value = opt_config.get('lr', 1e-4)
#     if isinstance(lr_value, str):
#         lr_value = float(lr_value)
    
#     # Handle weight decay - ensure it's a float  
#     wd_value = opt_config.get('weight_decay', 1e-4)
#     if isinstance(wd_value, str):
#         wd_value = float(wd_value)
    
#     optimizer_adapted = {
#         'type': opt_config.get('name', 'adam'),
#         'learning_rate': lr_value,
#         'weight_decay': wd_value,
#         'betas': opt_config.get('betas', [0.9, 0.999])
#     }
    
#     # Extract scheduler config with type conversion
#     sched_config = training_section.get('scheduler', {})
#     scheduler_adapted = {
#         'type': sched_config.get('name', 'cosine'),
#         'T_max': int(sched_config.get('T_max', 100)),
#         'eta_min': float(sched_config.get('eta_min', 1e-6)),
#         'step_size': int(sched_config.get('step_size', 30)),
#         'gamma': float(sched_config.get('gamma', 0.1)),
#         'patience': int(sched_config.get('patience', 10)),
#         'factor': float(sched_config.get('factor', 0.5))
#     }
    
#     # Extract loss config with type conversion
#     loss_config = training_section.get('loss', {})
#     loss_adapted = {
#         'loss_type': loss_config.get('name', 'combined'),
#         'dice_weight': float(loss_config.get('dice_weight', 0.5)),
#         'focal_weight': float(loss_config.get('focal_weight', 0.3)),
#         'focal_alpha': float(loss_config.get('focal_alpha', 0.25)),
#         'focal_gamma': float(loss_config.get('focal_gamma', 2.0)),
#         'aux_weight': 0.3
#     }
    
#     # Extract early stopping config with type conversion
#     es_config = training_section.get('early_stopping', {})
#     early_stopping_adapted = {
#         'patience': int(es_config.get('patience', 15)),
#         'min_delta': float(es_config.get('min_delta', 0.001)),
#         'mode': es_config.get('mode', 'max'),
#         'restore_best_weights': True
#     }
    
#     adapted = {
#         'epochs': int(training_section.get('epochs', 100)),
#         'batch_size': int(training_section.get('batch_size', 16)),
#         'num_workers': int(training_section.get('num_workers', 4)),
#         'mixed_precision': training_section.get('precision', 16) == 16,
#         'gradient_clipping': float(training_section.get('gradient_clip_val', 1.0)),
#         'log_interval': 10,
#         'validation_interval': 1,
#         'save_interval': 5,
#         'optimizer': optimizer_adapted,
#         'scheduler': scheduler_adapted,
#         'loss': loss_adapted,
#         'early_stopping': early_stopping_adapted
#     }
#     return adapted


# def create_optimizer(model: torch.nn.Module, config: dict) -> optim.Optimizer:
#     """Create optimizer from configuration."""
#     optimizer_config = config.get('optimizer', {})
#     optimizer_type = optimizer_config.get('type', 'adamw').lower()
#     learning_rate = optimizer_config.get('learning_rate', 1e-3)
#     weight_decay = optimizer_config.get('weight_decay', 1e-4)
    
#     if optimizer_type == 'adamw':
#         return optim.AdamW(
#             model.parameters(),
#             lr=learning_rate,
#             weight_decay=weight_decay,
#             betas=optimizer_config.get('betas', (0.9, 0.999))
#         )
#     elif optimizer_type == 'adam':
#         return optim.Adam(
#             model.parameters(),
#             lr=learning_rate,
#             weight_decay=weight_decay,
#             betas=optimizer_config.get('betas', (0.9, 0.999))
#         )
#     elif optimizer_type == 'sgd':
#         return optim.SGD(
#             model.parameters(),
#             lr=learning_rate,
#             weight_decay=weight_decay,
#             momentum=optimizer_config.get('momentum', 0.9)
#         )
#     else:
#         raise ValueError(f"Unknown optimizer type: {optimizer_type}")


# def print_model_info(model: torch.nn.Module, input_shape: tuple):
#     """Print model architecture information."""
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     print(f"\nModel Architecture Information:")
#     print(f"{'='*50}")
#     print(f"Total parameters: {total_params:,}")
#     print(f"Trainable parameters: {trainable_params:,}")
#     print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (float32)")
    
#     # Test forward pass to check model
#     try:
#         device = next(model.parameters()).device
#         dummy_input = torch.randn(1, *input_shape).to(device)
        
#         model.eval()
#         with torch.no_grad():
#             output = model(dummy_input)
#             if isinstance(output, dict):
#                 main_output_shape = output['out'].shape
#                 print(f"Input shape: (1, {', '.join(map(str, input_shape))})")
#                 print(f"Output shape: {main_output_shape}")
#                 if 'aux' in output:
#                     print(f"Auxiliary outputs: {len(output['aux'])}")
#             else:
#                 print(f"Input shape: (1, {', '.join(map(str, input_shape))})")
#                 print(f"Output shape: {output.shape}")
        
#         print("✅ Model architecture validated")
#     except Exception as e:
#         print(f"❌ Model validation failed: {e}")
    
#     print(f"{'='*50}")


# def print_training_summary(train_config: dict, data_config: dict, model_config: dict):
#     """Print training configuration summary."""
#     print(f"\nTraining Configuration Summary:")
#     print(f"{'='*60}")
    
#     print(f"Dataset:")
#     print(f"  - Data root: {data_config.get('data_root', 'N/A')}")
#     print(f"  - Batch size: {data_config.get('batch_size', train_config.get('batch_size', 'N/A'))}")
#     print(f"  - Image size: {data_config.get('image_size', 'N/A')}")
#     print(f"  - Augmentation: {data_config.get('augmentation', {}).get('enabled', False)}")
    
#     print(f"\nModel:")
#     print(f"  - Architecture: {model_config.get('architecture', 'N/A')}")
#     print(f"  - Backbone: {model_config.get('backbone', 'N/A')}")
#     print(f"  - Pretrained: {model_config.get('pretrained', 'N/A')}")
#     print(f"  - Input channels: {model_config.get('in_channels', 'N/A')}")
    
#     print(f"\nTraining:")
#     print(f"  - Epochs: {train_config.get('epochs', 'N/A')}")
#     print(f"  - Optimizer: {train_config.get('optimizer', {}).get('type', 'N/A')}")
#     print(f"  - Learning rate: {train_config.get('optimizer', {}).get('learning_rate', 'N/A')}")
#     print(f"  - Mixed precision: {train_config.get('mixed_precision', 'N/A')}")
#     print(f"  - Gradient clipping: {train_config.get('gradient_clipping', 'N/A')}")
    
#     print(f"\nLoss Function:")
#     loss_config = train_config.get('loss', {})
#     print(f"  - Type: {loss_config.get('loss_type', 'N/A')}")
#     print(f"  - Dice weight: {loss_config.get('dice_weight', 'N/A')}")
#     print(f"  - Focal weight: {loss_config.get('focal_weight', 'N/A')}")
#     print(f"  - Focal alpha: {loss_config.get('focal_alpha', 'N/A')}")
    
#     print(f"{'='*60}")


# def validate_config_compatibility(train_config: dict, data_config: dict, model_config: dict) -> bool:
#     """Validate that configurations have the expected nested structure."""
#     errors = []
    
#     # Check if configs have the expected nested structure
#     if 'training' not in train_config:
#         errors.append("Train config must have 'training' section")
#     if 'model' not in model_config:
#         errors.append("Model config must have 'model' section")
#     if 'data' not in data_config:
#         errors.append("Data config must have 'data' section")
    
#     # Check nested structure requirements
#     if 'training' in train_config:
#         training = train_config['training']
#         if 'epochs' not in training:
#             errors.append("Training section must specify 'epochs'")
#         if 'optimizer' not in training:
#             errors.append("Training section must have 'optimizer' subsection")
#         if 'loss' not in training:
#             errors.append("Training section must have 'loss' subsection")
    
#     if 'model' in model_config:
#         model = model_config['model']
#         if 'unet' not in model:
#             errors.append("Model section must have 'unet' subsection")
    
#     if 'data' in data_config:
#         data = data_config['data']
#         if 'tiles_dir' not in data:
#             errors.append("Data section must specify 'tiles_dir'")
    
#     if errors:
#         print("❌ Configuration validation failed:")
#         for error in errors:
#             print(f"   - {error}")
#         return False
    
#     print("✅ Configuration validation passed")
#     return True


# def main():
#     """Main training function."""
#     parser = argparse.ArgumentParser(description='Train forest fire detection model')
    
#     # Configuration files
#     parser.add_argument('--config', type=str, required=True,
#                        help='Path to training configuration file')
#     parser.add_argument('--data-config', type=str, required=True,
#                        help='Path to data configuration file')
#     parser.add_argument('--model-config', type=str, required=True,
#                        help='Path to model configuration file')
    
#     # Training options
#     parser.add_argument('--resume', type=str, default=None,
#                        help='Path to checkpoint to resume from')
#     parser.add_argument('--device', type=str, default='cuda',
#                        help='Device to use for training (cuda/cpu)')
#     parser.add_argument('--num-workers', type=int, default=4,
#                        help='Number of data loader workers')
#     parser.add_argument('--seed', type=int, default=42,
#                        help='Random seed for reproducibility')
    
#     # Output directories
#     parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
#                        help='Directory to save checkpoints')
#     parser.add_argument('--log-dir', type=str, default='logs',
#                        help='Directory to save logs')
    
#     # Debug options
#     parser.add_argument('--debug', action='store_true',
#                        help='Enable debug mode (small dataset)')
#     parser.add_argument('--profile', action='store_true',
#                        help='Enable profiling')
    
#     args = parser.parse_args()
    
#     # Set seed for reproducibility
#     set_seed(args.seed)
#     print(f"🌱 Set random seed to {args.seed}")
    
#     # Check device availability
#     if args.device == 'cuda' and not torch.cuda.is_available():
#         print("⚠️  CUDA not available, falling back to CPU")
#         args.device = 'cpu'
    
#     print(f"🖥️  Using device: {args.device}")
    
#     if args.device == 'cuda':
#         print(f"   GPU: {torch.cuda.get_device_name()}")
#         print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
#     # Load raw configurations
#     print("\n📋 Loading configurations...")
#     try:
#         train_config_raw = load_config(args.config)
#         data_config_raw = load_config(args.data_config)
#         model_config_raw = load_config(args.model_config)
#         print("✅ Configurations loaded successfully")
#     except Exception as e:
#         print(f"❌ Failed to load configurations: {e}")
#         return 1
    
#     # Validate raw configuration structure
#     if not validate_config_compatibility(train_config_raw, data_config_raw, model_config_raw):
#         return 1
    
#     # Adapt configurations to expected format
#     train_config = adapt_train_config(train_config_raw)
#     data_config = adapt_data_config(data_config_raw)
#     model_config = adapt_model_config(model_config_raw)
    
#     print("✅ Configurations adapted successfully")
    
#     # Print training summary
#     print_training_summary(train_config, data_config, model_config)
    
#     # Override config with command line arguments
#     if args.num_workers is not None:
#         train_config['num_workers'] = args.num_workers
#         data_config['num_workers'] = args.num_workers
    
#     # Create data loaders
#     print("\n📊 Creating data loaders...")
#     try:
#         train_loader, val_loader, test_loader = create_data_loaders(
#             data_config=data_config,
#             train_config=train_config,
#             num_workers=args.num_workers
#         )
        
#         print(f"✅ Data loaders created:")
#         print(f"   Training batches: {len(train_loader)}")
#         print(f"   Validation batches: {len(val_loader)}")
#         if test_loader:
#             print(f"   Test batches: {len(test_loader)}")
        
#         # Get sample batch for model validation
#         sample_batch = next(iter(train_loader))
#         sample_images, sample_masks = sample_batch
#         input_shape = sample_images.shape[1:]  # (C, H, W)
        
#         print(f"   Batch size: {sample_images.shape[0]}")
#         print(f"   Input shape: {input_shape}")
#         print(f"   Target shape: {sample_masks.shape}")
        
#     except Exception as e:
#         print(f"❌ Failed to create data loaders: {e}")
#         import traceback
#         traceback.print_exc()
#         return 1
    
#     # Create model
#     print("\n🏗️  Creating model...")
#     try:
#         model = create_model(model_config)
#         model = model.to(args.device)
#         print(f"✅ Model created: {model_config.get('architecture', 'Unknown')} with {model_config.get('backbone', 'Unknown')} backbone")
        
#         # Print model information
#         print_model_info(model, input_shape)
        
#     except Exception as e:
#         print(f"❌ Failed to create model: {e}")
#         import traceback
#         traceback.print_exc()
#         return 1
    
#     # Create loss function
#     print("\n🎯 Creating loss function...")
#     try:
#         loss_config = train_config.get('loss', {})
#         loss_fn = create_loss_function(loss_config)
#         print(f"✅ Loss function created: {loss_config.get('loss_type', 'combined')}")
        
#         if loss_config.get('loss_type') == 'combined':
#             print(f"   Dice weight: {loss_config.get('dice_weight', 0.5)}")
#             print(f"   Focal weight: {loss_config.get('focal_weight', 0.5)}")
#             print(f"   Focal alpha: {loss_config.get('focal_alpha', 0.75)}")
#             print(f"   Focal gamma: {loss_config.get('focal_gamma', 2.0)}")
        
#     except Exception as e:
#         print(f"❌ Failed to create loss function: {e}")
#         import traceback
#         traceback.print_exc()
#         return 1
    
#     # Create optimizer
#     print("\n⚙️  Creating optimizer...")
#     try:
#         optimizer = create_optimizer(model, train_config)
#         optimizer_config = train_config.get('optimizer', {})
#         print(f"✅ Optimizer created: {optimizer_config.get('type', 'adamw').upper()}")
#         print(f"   Learning rate: {optimizer_config.get('learning_rate', 1e-3)}")
#         print(f"   Weight decay: {optimizer_config.get('weight_decay', 1e-4)}")
        
#     except Exception as e:
#         print(f"❌ Failed to create optimizer: {e}")
#         import traceback
#         traceback.print_exc()
#         return 1
    
#     # Create trainer
#     print("\n🚀 Setting up trainer...")
#     try:
#         trainer = create_trainer(
#             model=model,
#             train_loader=train_loader,
#             val_loader=val_loader,
#             loss_fn=loss_fn,
#             optimizer=optimizer,
#             config=train_config,
#             device=args.device,
#             checkpoint_dir=args.checkpoint_dir,
#             log_dir=args.log_dir
#         )
#         print("✅ Trainer setup complete")
        
#         # Print training settings
#         print(f"   Epochs: {train_config.get('epochs', 100)}")
#         print(f"   Mixed precision: {train_config.get('mixed_precision', True)}")
#         print(f"   Gradient clipping: {train_config.get('gradient_clipping', 1.0)}")
        
#         if 'scheduler' in train_config:
#             scheduler_config = train_config['scheduler']
#             print(f"   LR scheduler: {scheduler_config.get('type', 'cosine')}")
        
#         if 'early_stopping' in train_config:
#             es_config = train_config['early_stopping']
#             print(f"   Early stopping: patience={es_config.get('patience', 10)}")
        
#     except Exception as e:
#         print(f"❌ Failed to setup trainer: {e}")
#         import traceback
#         traceback.print_exc()
#         return 1
    
#     # Debug mode: limit dataset size
#     if args.debug:
#         print("\n🐛 Debug mode enabled - limiting dataset size")
#         # Limit to first few batches for quick testing
#         train_loader.dataset.indices = train_loader.dataset.indices[:32] if hasattr(train_loader.dataset, 'indices') else None
#         val_loader.dataset.indices = val_loader.dataset.indices[:16] if hasattr(val_loader.dataset, 'indices') else None
#         train_config['epochs'] = min(train_config.get('epochs', 100), 3)  # Max 3 epochs in debug
    
#     # Start training
#     print(f"\n🔥 Starting training...")
#     print(f"{'='*80}")
    
#     try:
#         if args.profile:
#             # Enable profiling if requested
#             print("📊 Profiling enabled")
#             with torch.profiler.profile(
#                 schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#                 on_trace_ready=torch.profiler.tensorboard_trace_handler(args.log_dir),
#                 record_shapes=True,
#                 profile_memory=True,
#                 with_stack=True
#             ) as prof:
#                 training_results = trainer.train(resume_checkpoint=args.resume)
#                 prof.step()
#         else:
#             training_results = trainer.train(resume_checkpoint=args.resume)
        
#         # Training completed successfully
#         print(f"\n{'='*80}")
#         print("🎉 Training completed successfully!")
        
#         best_metric = training_results.get('best_val_metric', 0)
#         total_epochs = training_results.get('total_epochs', 0)
#         total_time = training_results.get('total_time', 0)
        
#         print(f"📊 Training Summary:")
#         print(f"   Best validation metric: {best_metric:.4f}")
#         print(f"   Total epochs: {total_epochs}")
#         print(f"   Total time: {total_time/3600:.2f} hours")
#         print(f"   Average time per epoch: {total_time/max(total_epochs, 1)/60:.1f} minutes")
        
#         # Save final summary
#         summary = {
#             'config': {
#                 'train': train_config,
#                 'data': data_config,
#                 'model': model_config
#             },
#             'results': training_results,
#             'args': vars(args)
#         }
        
#         summary_path = Path(args.log_dir) / 'training_summary.yaml'
#         summary_path.parent.mkdir(parents=True, exist_ok=True)
#         with open(summary_path, 'w') as f:
#             yaml.dump(summary, f, indent=2, default_flow_style=False)
        
#         print(f"💾 Training summary saved to: {summary_path}")
        
#         # Print next steps
#         print(f"\n🚀 Next Steps:")
#         print(f"   1. Check logs: {args.log_dir}")
#         print(f"   2. Best model: {args.checkpoint_dir}/best_checkpoint.pth")
#         print(f"   3. Run inference: python scripts/predict.py")
#         print(f"   4. Evaluate on test set")
        
#         return 0
        
#     except KeyboardInterrupt:
#         print(f"\n⚠️  Training interrupted by user")
#         return 1
#     except Exception as e:
#         print(f"\n❌ Training failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return 1


# if __name__ == "__main__":
#     """Entry point for training script."""
#     try:
#         exit_code = main()
#         sys.exit(exit_code)
#     except Exception as e:
#         print(f"💥 Unexpected error: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)

