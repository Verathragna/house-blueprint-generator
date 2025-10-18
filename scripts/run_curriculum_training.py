#!/usr/bin/env python3
"""
Run enhanced curriculum training with constraint-aware losses.

This script orchestrates:
1. Realistic data generation
2. Multi-stage curriculum training  
3. Constraint-aware loss functions
4. Physics-informed training

Usage:
    python scripts/run_curriculum_training.py [options]
"""
import argparse
import logging
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.layout_transformer import LayoutTransformer
from tokenizer.tokenizer import BlueprintTokenizer
from training.curriculum_training import CurriculumTrainer

def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('curriculum_training.log')
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='Run curriculum training')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--checkpoint_dir', default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--data_dir', default='dataset', help='Dataset directory')
    parser.add_argument('--resume', help='Resume from checkpoint')
    parser.add_argument('--generate_data', action='store_true', 
                       help='Generate realistic training data first')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting enhanced curriculum training...")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Generate realistic training data if requested
    if args.generate_data:
        logger.info("Generating realistic training data...")
        from dataset.realistic_layouts import generate_realistic_dataset
        
        realistic_train_path = os.path.join(args.data_dir, 'realistic_train.jsonl')
        realistic_val_path = os.path.join(args.data_dir, 'realistic_val.jsonl')
        
        # Generate training and validation sets
        generate_realistic_dataset(1500, realistic_train_path, seed=42)
        generate_realistic_dataset(300, realistic_val_path, seed=24)
        
        logger.info(f"Generated realistic datasets:")
        logger.info(f"  Train: {realistic_train_path}")
        logger.info(f"  Val: {realistic_val_path}")
    
    # Setup model and tokenizer
    logger.info("Initializing model and tokenizer...")
    tokenizer = BlueprintTokenizer()
    vocab_size = tokenizer.get_vocab_size()
    
    model = LayoutTransformer(
        vocab_size=vocab_size,
        d_model=256,
        num_layers=6,
        dim_ff=1024,
        nhead=8
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
    
    # Move model to device
    import torch
    device = torch.device(args.device)
    model.to(device)
    
    # Setup curriculum trainer
    trainer = CurriculumTrainer(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Data paths
    train_path = os.path.join(args.data_dir, 'train.jsonl')
    val_path = os.path.join(args.data_dir, 'val.jsonl')
    
    # Check for realistic data paths
    realistic_train_path = os.path.join(args.data_dir, 'realistic_train.jsonl')
    realistic_val_path = os.path.join(args.data_dir, 'realistic_val.jsonl')
    
    # Use realistic data if available
    if os.path.exists(realistic_train_path):
        train_path = realistic_train_path
        logger.info("Using realistic training data")
    
    if os.path.exists(realistic_val_path):
        val_path = realistic_val_path
        logger.info("Using realistic validation data")
    
    # Verify data files exist
    if not os.path.exists(train_path):
        logger.error(f"Training data not found: {train_path}")
        logger.info("Run with --generate_data to create realistic training data")
        return 1
    
    if not os.path.exists(val_path):
        logger.warning(f"Validation data not found: {val_path}, using training data")
        val_path = train_path
    
    logger.info(f"Training data: {train_path}")
    logger.info(f"Validation data: {val_path}")
    
    # Run curriculum training
    try:
        trainer.train_curriculum(
            train_path=train_path,
            val_path=val_path,
            batch_size=args.batch_size
        )
        
        logger.info("Curriculum training completed successfully!")
        
        # Save final model
        final_path = os.path.join(args.checkpoint_dir, 'curriculum_final.pth')
        torch.save(model.state_dict(), final_path)
        logger.info(f"Final model saved to: {final_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())