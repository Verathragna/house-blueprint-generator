"""
Multi-stage curriculum training for layout generation.

Stage 1: Simple layouts (2-4 rooms, low density)
Stage 2: Medium layouts (4-6 rooms, moderate density)  
Stage 3: Complex layouts (6+ rooms, higher density)

Each stage progressively increases complexity while maintaining quality.
"""
import os
import torch
import logging
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, Subset
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CurriculumStage:
    """Configuration for a curriculum training stage."""
    name: str
    max_rooms: int
    max_density: float
    min_samples: int
    epochs: int
    learning_rate: float
    constraint_weight: float

class CurriculumDataset(Dataset):
    """Dataset that can filter samples by curriculum stage criteria."""
    
    def __init__(self, jsonl_path: str, tokenizer, stage: Optional[CurriculumStage] = None):
        self.tokenizer = tokenizer
        self.stage = stage
        
        # Load all samples
        all_samples = []
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        sample = json.loads(line)
                        all_samples.append(sample)
                    except json.JSONDecodeError:
                        continue
        
        # Filter samples based on curriculum stage
        if stage:
            self.samples = self._filter_by_stage(all_samples, stage)
            logger.info(f"Stage '{stage.name}': {len(self.samples)} samples (filtered from {len(all_samples)})")
        else:
            self.samples = all_samples
            logger.info(f"All samples: {len(self.samples)}")
    
    def _filter_by_stage(self, samples: List[Dict], stage: CurriculumStage) -> List[Dict]:
        """Filter samples that match the curriculum stage criteria."""
        filtered = []
        
        for sample in samples:
            # Analyze layout complexity
            layout = sample.get("layout", {})
            rooms = layout.get("layout", {}).get("rooms", [])
            
            # Room count check
            if len(rooms) > stage.max_rooms:
                continue
                
            # Density check
            density = self._calculate_density(rooms)
            if density > stage.max_density:
                continue
                
            # Quality check - skip layouts with obvious issues
            if self._has_obvious_issues(rooms):
                continue
                
            filtered.append(sample)
            
            # Stop when we have enough samples for this stage
            if len(filtered) >= stage.min_samples * 2:  # Get extra samples for variety
                break
                
        return filtered[:stage.min_samples] if len(filtered) > stage.min_samples else filtered
    
    def _calculate_density(self, rooms: List[Dict]) -> float:
        """Calculate area density of layout."""
        if not rooms:
            return 0.0
            
        total_area = sum(
            float(room.get("size", {}).get("width", 0)) *
            float(room.get("size", {}).get("length", 0))
            for room in rooms
        )
        
        # Assume 40x40 layout bounds
        available_area = 40.0 * 40.0
        return total_area / available_area if available_area > 0 else 0.0
    
    def _has_obvious_issues(self, rooms: List[Dict]) -> bool:
        """Check for obvious layout issues."""
        if not rooms:
            return True
            
        # Check for rooms outside bounds
        for room in rooms:
            x = float(room.get("position", {}).get("x", 0))
            y = float(room.get("position", {}).get("y", 0))
            w = float(room.get("size", {}).get("width", 0))
            h = float(room.get("size", {}).get("length", 0))
            
            if x < 0 or y < 0 or x + w > 40 or y + h > 40:
                return True
                
            if w <= 0 or h <= 0:
                return True
        
        # Check for severe overlaps
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                if self._rooms_severely_overlap(rooms[i], rooms[j]):
                    return True
                    
        return False
    
    def _rooms_severely_overlap(self, room1: Dict, room2: Dict) -> bool:
        """Check if rooms overlap by more than 50% of either room's area."""
        x1 = float(room1.get("position", {}).get("x", 0))
        y1 = float(room1.get("position", {}).get("y", 0))
        w1 = float(room1.get("size", {}).get("width", 0))
        h1 = float(room1.get("size", {}).get("length", 0))
        
        x2 = float(room2.get("position", {}).get("x", 0))
        y2 = float(room2.get("position", {}).get("y", 0))
        w2 = float(room2.get("size", {}).get("width", 0))
        h2 = float(room2.get("size", {}).get("length", 0))
        
        # Calculate overlap area
        overlap_w = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        overlap_h = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        overlap_area = overlap_w * overlap_h
        
        area1 = w1 * h1
        area2 = w2 * h2
        
        # Check if overlap is more than 50% of either room
        return overlap_area > 0.5 * min(area1, area2)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Check if already tokenized
        if "x" in sample and "y" in sample:
            return (
                torch.tensor(sample["x"], dtype=torch.long),
                torch.tensor(sample["y"], dtype=torch.long),
            )
        
        # Tokenize on the fly
        x_ids, y_ids = self.tokenizer.build_training_pair(
            sample["params"], sample["layout"]
        )
        return torch.tensor(x_ids, dtype=torch.long), torch.tensor(y_ids, dtype=torch.long)


class CurriculumTrainer:
    """Manages multi-stage curriculum training."""
    
    def __init__(self, model, tokenizer, device="cpu", checkpoint_dir="checkpoints"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.checkpoint_dir = checkpoint_dir
        
        # Define curriculum stages
        self.stages = [
            CurriculumStage(
                name="Simple",
                max_rooms=4,
                max_density=0.4,
                min_samples=800,
                epochs=15,
                learning_rate=3e-4,
                constraint_weight=0.5
            ),
            CurriculumStage(
                name="Medium", 
                max_rooms=6,
                max_density=0.55,
                min_samples=600,
                epochs=12,
                learning_rate=2e-4,
                constraint_weight=1.0
            ),
            CurriculumStage(
                name="Complex",
                max_rooms=8,
                max_density=0.7,
                min_samples=400,
                epochs=10,
                learning_rate=1e-4,
                constraint_weight=1.5
            )
        ]
    
    def train_curriculum(self, train_path: str, val_path: str, 
                        batch_size: int = 16) -> None:
        """Execute full curriculum training."""
        logger.info("Starting curriculum training...")
        
        # Add realistic layout data first
        self._prepare_realistic_data(train_path)
        
        for stage_idx, stage in enumerate(self.stages):
            logger.info(f"\\n{'='*50}")
            logger.info(f"STAGE {stage_idx + 1}: {stage.name}")
            logger.info(f"Max rooms: {stage.max_rooms}, Max density: {stage.max_density:.1%}")
            logger.info(f"{'='*50}")
            
            # Create stage-specific datasets
            train_dataset = CurriculumDataset(train_path, self.tokenizer, stage)
            val_dataset = CurriculumDataset(val_path, self.tokenizer, stage)
            
            if len(train_dataset) < 50:  # Minimum viable training set
                logger.warning(f"Stage {stage.name} has too few samples ({len(train_dataset)}), skipping...")
                continue
            
            # Create data loaders
            from training.train import collate_fn
            train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                    shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                  shuffle=False, collate_fn=collate_fn)
            
            # Train this stage
            self._train_stage(stage, train_loader, val_loader, stage_idx)
            
            logger.info(f"Completed stage {stage.name}")
    
    def _prepare_realistic_data(self, train_path: str) -> None:
        """Generate and add realistic layout data to training set."""
        realistic_path = train_path.replace('.jsonl', '_realistic.jsonl')
        
        if not os.path.exists(realistic_path):
            logger.info("Generating realistic layout dataset...")
            from dataset.realistic_layouts import generate_realistic_dataset
            generate_realistic_dataset(1000, realistic_path, seed=42)
        
        # Merge realistic data with existing training data
        merged_path = train_path.replace('.jsonl', '_merged.jsonl')
        
        with open(merged_path, 'w', encoding='utf-8') as out_f:
            # Copy original data
            if os.path.exists(train_path):
                with open(train_path, 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        out_f.write(line)
            
            # Add realistic data
            with open(realistic_path, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    out_f.write(line)
        
        # Replace original with merged
        os.replace(merged_path, train_path)
        logger.info("Added realistic layouts to training data")
    
    def _train_stage(self, stage: CurriculumStage, train_loader: DataLoader,
                    val_loader: DataLoader, stage_idx: int) -> None:
        """Train a single curriculum stage."""
        from training.constraint_losses import ConstraintLoss
        from tokenizer.tokenizer import PAD_ID
        
        # Setup optimizer for this stage
        optimizer = torch.optim.AdamW(self.model.parameters(), 
                                    lr=stage.learning_rate, weight_decay=0.01)
        
        # Setup loss functions
        base_criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
        constraint_loss = ConstraintLoss(
            area_weight=stage.constraint_weight * 0.5,
            overlap_weight=stage.constraint_weight * 1.0,
            boundary_weight=stage.constraint_weight * 0.8,
            count_weight=stage.constraint_weight * 0.3
        )
        
        # Training loop
        for epoch in range(stage.epochs):
            self.model.train()
            total_loss = 0.0
            constraint_total = 0.0
            
            for batch_idx, (x, y, mask) in enumerate(train_loader):
                x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = self.model(x, key_padding_mask=mask)
                    base_loss = base_criterion(logits.reshape(-1, self.tokenizer.get_vocab_size()), 
                                             y.reshape(-1))
                    
                    # Add constraint loss
                    total_loss_tensor, loss_dict = constraint_loss(
                        logits, y, self.tokenizer, base_loss
                    )
                
                # Backward pass
                total_loss_tensor.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += total_loss_tensor.item()
                constraint_total += loss_dict.get('constraint_loss', 0.0)
                
                if batch_idx % 20 == 0:
                    logger.info(f"Stage {stage.name} Epoch {epoch+1}/{stage.epochs} "
                              f"Batch {batch_idx}: Loss={total_loss_tensor.item():.4f} "
                              f"(base={loss_dict['base_loss']:.4f}, "
                              f"constraint={loss_dict['constraint_loss']:.4f})")
            
            # Validation
            val_loss = self._validate(val_loader, base_criterion, constraint_loss)
            
            avg_train_loss = total_loss / len(train_loader)
            avg_constraint = constraint_total / len(train_loader)
            
            logger.info(f"Stage {stage.name} Epoch {epoch+1}: "
                       f"Train={avg_train_loss:.4f} (constraint={avg_constraint:.4f}), "
                       f"Val={val_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f"curriculum_stage{stage_idx+1}_{stage.name.lower()}_epoch{epoch+1}.pt"
            )
            torch.save({
                "model": self.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "stage": stage_idx,
                "epoch": epoch,
                "stage_name": stage.name
            }, checkpoint_path)
        
        # Save final stage checkpoint
        final_path = os.path.join(self.checkpoint_dir, f"stage_{stage.name.lower()}_final.pt")
        torch.save(self.model.state_dict(), final_path)
        logger.info(f"Saved final checkpoint for stage {stage.name}")
    
    def _validate(self, val_loader: DataLoader, base_criterion, constraint_loss) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
                
                logits = self.model(x, key_padding_mask=mask)
                base_loss = base_criterion(logits.reshape(-1, self.tokenizer.get_vocab_size()),
                                         y.reshape(-1))
                
                total_loss_tensor, _ = constraint_loss(logits, y, self.tokenizer, base_loss)
                total_loss += total_loss_tensor.item()
        
        return total_loss / len(val_loader) if len(val_loader) > 0 else 0.0


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    
    from models.layout_transformer import LayoutTransformer
    from tokenizer.tokenizer import BlueprintTokenizer
    
    # Setup
    tokenizer = BlueprintTokenizer()
    model = LayoutTransformer(tokenizer.get_vocab_size())
    trainer = CurriculumTrainer(model, tokenizer)
    
    # Run curriculum training
    trainer.train_curriculum("dataset/train.jsonl", "dataset/val.jsonl")