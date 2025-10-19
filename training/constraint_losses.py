"""
Constraint-aware loss functions for layout generation.

These losses penalize layouts that violate physical constraints like:
- Area density thresholds
- Room overlaps
- Boundary violations  
- Unrealistic room counts
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ConstraintLoss:
    """Container for various constraint-based loss components."""
    
    def __init__(self, 
                 area_weight: float = 1.0,
                 overlap_weight: float = 2.0, 
                 boundary_weight: float = 1.5,
                 count_weight: float = 0.5,
                 max_width: float = 40.0,
                 max_height: float = 40.0):
        """
        Args:
            area_weight: Weight for area density penalty
            overlap_weight: Weight for room overlap penalty
            boundary_weight: Weight for boundary violation penalty
            count_weight: Weight for excessive room count penalty
            max_width: Maximum layout width
            max_height: Maximum layout height
        """
        self.area_weight = area_weight
        self.overlap_weight = overlap_weight
        self.boundary_weight = boundary_weight
        self.count_weight = count_weight
        self.max_width = max_width
        self.max_height = max_height
        
    def __call__(self, 
                 logits: torch.Tensor,
                 targets: torch.Tensor, 
                 tokenizer,
                 base_loss: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute constraint-aware loss.
        
        Args:
            logits: Model predictions [batch_size, seq_len, vocab_size]
            targets: Target tokens [batch_size, seq_len]  
            tokenizer: BlueprintTokenizer for decoding
            base_loss: Base cross-entropy loss
            
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        batch_size = targets.size(0)
        device = targets.device
        
        # Decode layouts from target tokens for constraint evaluation
        constraint_losses = []
        loss_dict = {}
        
        for i in range(batch_size):
            target_tokens = targets[i].cpu().tolist()
            try:
                # Remove padding and decode
                target_tokens = [t for t in target_tokens if t != tokenizer.pad_token_id]
                layout = tokenizer.decode_layout_tokens(target_tokens)
                
                # Calculate constraint violations
                area_loss = self._area_density_loss(layout)
                overlap_loss = self._overlap_loss(layout)
                boundary_loss = self._boundary_loss(layout)
                count_loss = self._room_count_loss(layout)
                
                architectural_loss = self._architectural_loss(layout)
                
                sample_constraint_loss = (
                    self.area_weight * area_loss +
                    self.overlap_weight * overlap_loss + 
                    self.boundary_weight * boundary_loss +
                    self.count_weight * count_loss +
                    0.5 * architectural_loss  # Weight for architectural rules
                )
                
                constraint_losses.append(sample_constraint_loss)
                
            except Exception as e:
                # If decode fails, use zero constraint loss for this sample
                logger.debug(f"Failed to decode layout for constraint loss: {e}")
                constraint_losses.append(0.0)
        
        # Convert to tensor and compute mean
        if constraint_losses:
            constraint_loss_tensor = torch.tensor(constraint_losses, device=device).mean()
        else:
            constraint_loss_tensor = torch.tensor(0.0, device=device)
            
        total_loss = base_loss + constraint_loss_tensor
        
        loss_dict = {
            'base_loss': base_loss.item(),
            'constraint_loss': constraint_loss_tensor.item(), 
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def _area_density_loss(self, layout: Dict) -> float:
        """Penalize layouts with too high area density."""
        rooms = layout.get("layout", {}).get("rooms", [])
        if not rooms:
            return 0.0
            
        total_room_area = sum(
            float(room.get("size", {}).get("width", 0)) *
            float(room.get("size", {}).get("length", 0))
            for room in rooms
        )
        
        available_area = self.max_width * self.max_height
        density_ratio = total_room_area / available_area if available_area > 0 else 0.0
        
        # Penalize densities above 0.6 (60% of space)
        if density_ratio > 0.6:
            return (density_ratio - 0.6) ** 2
        return 0.0
    
    def _overlap_loss(self, layout: Dict) -> float:
        """Penalize overlapping rooms."""
        rooms = layout.get("layout", {}).get("rooms", [])
        if len(rooms) < 2:
            return 0.0
            
        overlap_penalty = 0.0
        
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                room1, room2 = rooms[i], rooms[j]
                
                # Get room bounds
                x1 = float(room1.get("position", {}).get("x", 0))
                y1 = float(room1.get("position", {}).get("y", 0))
                w1 = float(room1.get("size", {}).get("width", 0))
                h1 = float(room1.get("size", {}).get("length", 0))
                
                x2 = float(room2.get("position", {}).get("x", 0))
                y2 = float(room2.get("position", {}).get("y", 0))
                w2 = float(room2.get("size", {}).get("width", 0))
                h2 = float(room2.get("size", {}).get("length", 0))
                
                # Check for overlap
                if (x1 < x2 + w2 and x1 + w1 > x2 and 
                    y1 < y2 + h2 and y1 + h1 > y2):
                    # Calculate overlap area
                    overlap_w = min(x1 + w1, x2 + w2) - max(x1, x2)
                    overlap_h = min(y1 + h1, y2 + h2) - max(y1, y2)
                    overlap_area = max(0, overlap_w) * max(0, overlap_h)
                    overlap_penalty += overlap_area / max(w1 * h1, w2 * h2, 1.0)
        
        return overlap_penalty
    
    def _boundary_loss(self, layout: Dict) -> float:
        """Penalize rooms that exceed layout boundaries.""" 
        rooms = layout.get("layout", {}).get("rooms", [])
        if not rooms:
            return 0.0
            
        boundary_penalty = 0.0
        
        for room in rooms:
            x = float(room.get("position", {}).get("x", 0))
            y = float(room.get("position", {}).get("y", 0))
            w = float(room.get("size", {}).get("width", 0))
            h = float(room.get("size", {}).get("length", 0))
            
            # Penalize negative positions
            if x < 0:
                boundary_penalty += abs(x)
            if y < 0:
                boundary_penalty += abs(y)
                
            # Penalize exceeding max bounds
            if x + w > self.max_width:
                boundary_penalty += (x + w - self.max_width)
            if y + h > self.max_height:
                boundary_penalty += (y + h - self.max_height)
                
        return boundary_penalty
    
    def _room_count_loss(self, layout: Dict) -> float:
        """Penalize layouts with too many rooms."""
        rooms = layout.get("layout", {}).get("rooms", [])
        room_count = len(rooms)
        
        # Penalize more than 8 rooms (gets exponentially expensive)
        if room_count > 8:
            return (room_count - 8) ** 1.5
        return 0.0
    
    def _architectural_loss(self, layout: Dict) -> float:
        """Penalize architectural rule violations."""
        rooms = layout.get("layout", {}).get("rooms", [])
        if not rooms:
            return 0.0
            
        penalty = 0.0
        
        # 1. Garage accessibility penalty
        garage_penalty = self._garage_access_penalty(rooms)
        
        # 2. Adjacency rule penalty
        adjacency_penalty = self._adjacency_penalty(rooms)
        
        # 3. Useless hallway penalty  
        hallway_penalty = self._useless_hallway_penalty(rooms)
        
        # 4. Room size penalty
        size_penalty = self._room_size_penalty(rooms)
        
        # 5. Entry access penalty
        entry_penalty = self._entry_access_penalty(rooms)
        
        # 6. Natural light penalty
        light_penalty = self._natural_light_penalty(rooms)
        
        penalty = (garage_penalty + adjacency_penalty + hallway_penalty + 
                  size_penalty + entry_penalty + light_penalty)
        return penalty
    
    def _garage_access_penalty(self, rooms: List[Dict]) -> float:
        """Penalize garages that don't touch boundary."""
        penalty = 0.0
        
        for room in rooms:
            if "garage" in room.get("type", "").lower():
                x = float(room.get("position", {}).get("x", 0))
                y = float(room.get("position", {}).get("y", 0))
                w = float(room.get("size", {}).get("width", 0))
                h = float(room.get("size", {}).get("length", 0))
                
                # Check if garage touches boundary
                touches_boundary = (
                    x <= 1.0 or y <= 1.0 or 
                    x + w >= self.max_width - 1.0 or 
                    y + h >= self.max_height - 1.0
                )
                
                if not touches_boundary:
                    penalty += 1.0  # Strong penalty for inaccessible garage
                    
        return penalty
    
    def _adjacency_penalty(self, rooms: List[Dict]) -> float:
        """Penalize poor room adjacency choices."""
        penalty = 0.0
        
        # Create room position lookup
        room_centers = {}
        for room in rooms:
            room_type = room.get("type", "").lower()
            x = float(room.get("position", {}).get("x", 0))
            y = float(room.get("position", {}).get("y", 0))
            w = float(room.get("size", {}).get("width", 0))
            h = float(room.get("size", {}).get("length", 0))
            
            center_x = x + w / 2
            center_y = y + h / 2
            
            if room_type not in room_centers:
                room_centers[room_type] = []
            room_centers[room_type].append((center_x, center_y))
        
        # Check kitchen-dining adjacency
        if "kitchen" in room_centers and "dining room" in room_centers:
            min_dist = float('inf')
            for kx, ky in room_centers["kitchen"]:
                for dx, dy in room_centers["dining room"]:
                    dist = ((kx - dx) ** 2 + (ky - dy) ** 2) ** 0.5
                    min_dist = min(min_dist, dist)
            
            if min_dist > 18.0:  # More than 18 feet apart
                penalty += 0.5
        
        return penalty
    
    def _useless_hallway_penalty(self, rooms: List[Dict]) -> float:
        """Penalize unnecessary or too-small hallways."""
        penalty = 0.0
        
        for room in rooms:
            if "hallway" in room.get("type", "").lower():
                w = float(room.get("size", {}).get("width", 0))
                h = float(room.get("size", {}).get("length", 0))
                area = w * h
                
                # Penalize very small hallways
                if area < 20.0:
                    penalty += 0.3
                    
                # Penalize hallways when there are few rooms
                if len(rooms) <= 4:
                    penalty += 0.2
                    
        return penalty
    
    def _room_size_penalty(self, rooms: List[Dict]) -> float:
        """Penalize rooms with unreasonable sizes."""
        penalty = 0.0
        
        min_sizes = {
            "bedroom": 81, "bathroom": 30, "kitchen": 64, "living room": 144,
            "dining room": 80, "garage": 240, "laundry room": 30, "office": 64
        }
        
        for room in rooms:
            room_type = room.get("type", "").lower()
            w = float(room.get("size", {}).get("width", 0))
            h = float(room.get("size", {}).get("length", 0))
            area = w * h
            
            # Check against minimum sizes
            for room_key, min_area in min_sizes.items():
                if room_key in room_type:
                    if area < min_area:
                        penalty += (min_area - area) / min_area  # Normalized penalty
                    break
                    
        return penalty
    
    def _entry_access_penalty(self, rooms: List[Dict]) -> float:
        """Penalize layouts without proper entry access."""
        penalty = 0.0
        
        # Check if living room touches boundary
        living_room_accessible = False
        for room in rooms:
            if "living room" in room.get("type", "").lower():
                x = float(room.get("position", {}).get("x", 0))
                y = float(room.get("position", {}).get("y", 0))
                w = float(room.get("size", {}).get("width", 0))
                h = float(room.get("size", {}).get("length", 0))
                
                touches_boundary = (
                    x <= 1.0 or y <= 1.0 or 
                    x + w >= self.max_width - 1.0 or 
                    y + h >= self.max_height - 1.0
                )
                
                if touches_boundary:
                    living_room_accessible = True
                    break
        
        if not living_room_accessible:
            penalty += 0.5  # Moderate penalty for no main entrance
            
        return penalty
    
    def _natural_light_penalty(self, rooms: List[Dict]) -> float:
        """Penalize rooms without access to exterior walls."""
        penalty = 0.0
        
        priority_rooms = ["living room", "kitchen", "bedroom", "dining room"]
        
        for room in rooms:
            room_type = room.get("type", "").lower()
            if any(priority in room_type for priority in priority_rooms):
                x = float(room.get("position", {}).get("x", 0))
                y = float(room.get("position", {}).get("y", 0))
                w = float(room.get("size", {}).get("width", 0))
                h = float(room.get("size", {}).get("length", 0))
                
                touches_exterior = (
                    x <= 1.0 or y <= 1.0 or 
                    x + w >= self.max_width - 1.0 or 
                    y + h >= self.max_height - 1.0
                )
                
                if not touches_exterior:
                    penalty += 0.2  # Small penalty per room without light
        
        return penalty


class PhysicsInformedLoss:
    """Physics-informed loss that considers spatial relationships."""
    
    def __init__(self, 
                 separation_weight: float = 1.0,
                 connectivity_weight: float = 0.5,
                 min_separation: float = 1.0):
        self.separation_weight = separation_weight
        self.connectivity_weight = connectivity_weight
        self.min_separation = min_separation
    
    def __call__(self, layout: Dict) -> float:
        """Calculate physics-informed loss for a layout."""
        rooms = layout.get("layout", {}).get("rooms", [])
        if len(rooms) < 2:
            return 0.0
            
        separation_loss = self._separation_loss(rooms)
        connectivity_loss = self._connectivity_loss(rooms)
        
        return (self.separation_weight * separation_loss + 
                self.connectivity_weight * connectivity_loss)
    
    def _separation_loss(self, rooms: List[Dict]) -> float:
        """Penalize rooms that are too close together."""
        penalty = 0.0
        
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                room1, room2 = rooms[i], rooms[j]
                
                # Calculate minimum distance between rooms
                dist = self._room_distance(room1, room2)
                if dist < self.min_separation:
                    penalty += (self.min_separation - dist) ** 2
                    
        return penalty / max(1, len(rooms) * (len(rooms) - 1) / 2)
    
    def _connectivity_loss(self, rooms: List[Dict]) -> float:
        """Penalize layouts where rooms are too isolated."""
        if len(rooms) < 2:
            return 0.0
            
        # Count rooms that share walls with others
        connected_rooms = 0
        
        for i, room1 in enumerate(rooms):
            for j, room2 in enumerate(rooms):
                if i != j and self._rooms_share_wall(room1, room2):
                    connected_rooms += 1
                    break
                    
        connectivity_ratio = connected_rooms / len(rooms)
        
        # Penalize if less than 80% of rooms are connected
        if connectivity_ratio < 0.8:
            return (0.8 - connectivity_ratio) ** 2
        return 0.0
    
    def _room_distance(self, room1: Dict, room2: Dict) -> float:
        """Calculate minimum distance between two rooms."""
        x1 = float(room1.get("position", {}).get("x", 0))
        y1 = float(room1.get("position", {}).get("y", 0))
        w1 = float(room1.get("size", {}).get("width", 0))
        h1 = float(room1.get("size", {}).get("length", 0))
        
        x2 = float(room2.get("position", {}).get("x", 0))
        y2 = float(room2.get("position", {}).get("y", 0))
        w2 = float(room2.get("size", {}).get("width", 0))
        h2 = float(room2.get("size", {}).get("length", 0))
        
        # Calculate distance between room rectangles
        dx = max(0, max(x1 - (x2 + w2), x2 - (x1 + w1)))
        dy = max(0, max(y1 - (y2 + h2), y2 - (y1 + h1)))
        
        return (dx ** 2 + dy ** 2) ** 0.5
    
    def _rooms_share_wall(self, room1: Dict, room2: Dict, tol: float = 1e-6) -> bool:
        """Check if two rooms share a wall segment."""
        x1 = float(room1.get("position", {}).get("x", 0))
        y1 = float(room1.get("position", {}).get("y", 0))
        w1 = float(room1.get("size", {}).get("width", 0))
        h1 = float(room1.get("size", {}).get("length", 0))
        
        x2 = float(room2.get("position", {}).get("x", 0))
        y2 = float(room2.get("position", {}).get("y", 0))
        w2 = float(room2.get("size", {}).get("width", 0))
        h2 = float(room2.get("size", {}).get("length", 0))
        
        # Check for shared vertical or horizontal walls
        vertical_touch = (abs((x1 + w1) - x2) < tol or abs((x2 + w2) - x1) < tol) and (
            min(y1 + h1, y2 + h2) - max(y1, y2) > 0
        )
        horizontal_touch = (abs((y1 + h1) - y2) < tol or abs((y2 + h2) - y1) < tol) and (
            min(x1 + w1, x2 + w2) - max(x1, x2) > 0
        )
        
        return vertical_touch or horizontal_touch