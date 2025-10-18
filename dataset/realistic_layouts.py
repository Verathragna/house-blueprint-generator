"""
Generate well-spaced, realistic layouts for training data augmentation.

Focus on creating layouts that satisfy physical constraints:
- Conservative area density (40-60% fill)
- Proper room separation
- Realistic room counts and sizes
- Good connectivity patterns
"""
import json
import random
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Room type priorities and typical sizes
ROOM_PRIORITIES = {
    "Living Room": 1, "Kitchen": 2, "Bedroom": 3, "Bathroom": 4,
    "Dining Room": 5, "Garage": 6, "Laundry Room": 7, "Office": 8,
    "Closet": 9, "Hallway": 10
}

# Conservative room sizes (width, length) in feet
REALISTIC_ROOM_SIZES = {
    "Living Room": [(14, 16), (16, 18), (18, 20)],
    "Kitchen": [(10, 12), (12, 14), (12, 16)], 
    "Bedroom": [(10, 12), (12, 12), (12, 14)],
    "Bathroom": [(6, 8), (8, 8), (8, 10)],
    "Dining Room": [(10, 12), (12, 12), (12, 14)],
    "Garage": [(20, 20), (20, 24), (24, 24)],
    "Laundry Room": [(6, 8), (8, 8), (8, 10)],
    "Office": [(10, 10), (10, 12), (12, 12)],
    "Closet": [(4, 6), (6, 6), (6, 8)],
    "Hallway": [(4, 8), (4, 12), (6, 12)]
}

class RealisticLayoutGenerator:
    """Generate realistic, well-spaced layouts for training."""
    
    def __init__(self, max_width: float = 40.0, max_height: float = 40.0,
                 target_density: float = 0.5, min_spacing: float = 2.0):
        self.max_width = max_width
        self.max_height = max_height  
        self.target_density = target_density
        self.min_spacing = min_spacing
        self.available_area = max_width * max_height
        
    def generate_layout(self, params: Dict, rng: random.Random = None) -> Dict:
        """Generate a realistic layout from parameters."""
        if rng is None:
            rng = random.Random()
            
        # Extract room requirements from parameters
        room_counts = self._extract_room_counts(params)
        
        # Generate room list with priorities
        room_types = self._generate_room_list(room_counts, rng)
        
        # Choose layout strategy based on room count and area
        total_rooms = sum(room_counts.values())
        
        if total_rooms <= 4:
            return self._generate_simple_grid(room_types, rng)
        elif total_rooms <= 6:
            return self._generate_l_shape(room_types, rng) 
        elif total_rooms <= 8:
            return self._generate_corridor_layout(room_types, rng)
        else:
            # Too many rooms - use emergency simplification
            return self._generate_compact_grid(room_types[:6], rng)  # Limit to 6 rooms
    
    def _extract_room_counts(self, params: Dict) -> Dict[str, int]:
        """Extract room counts from parameters."""
        counts = {}
        
        # Required rooms
        counts["Bedroom"] = max(1, int(params.get("bedrooms", 2)))
        counts["Bathroom"] = max(1, int(params.get("bathrooms", {}).get("full", 1)))
        counts["Kitchen"] = 1
        counts["Living Room"] = 1
        
        # Optional rooms based on size/style
        square_feet = int(params.get("squareFeet", 1500))
        
        if square_feet >= 1800:
            counts["Dining Room"] = 1
            
        if square_feet >= 2200:
            counts["Laundry Room"] = 1
            
        # Garage if specified
        garage_info = params.get("garage", {})
        if garage_info.get("attached", False):
            counts["Garage"] = 1
            
        return counts
    
    def _generate_room_list(self, room_counts: Dict[str, int], rng: random.Random) -> List[str]:
        """Generate prioritized list of room types."""
        room_list = []
        
        for room_type, count in room_counts.items():
            room_list.extend([room_type] * count)
            
        # Sort by priority
        room_list.sort(key=lambda r: ROOM_PRIORITIES.get(r, 99))
        return room_list
    
    def _generate_simple_grid(self, room_types: List[str], rng: random.Random) -> Dict:
        """Generate simple 2x2 grid layout for few rooms."""
        rooms = []
        
        # Calculate grid dimensions
        n_rooms = len(room_types)
        if n_rooms <= 2:
            cols, rows = 2, 1
        elif n_rooms <= 4:
            cols, rows = 2, 2
        else:
            cols, rows = 3, 2
            
        cell_w = (self.max_width - self.min_spacing * (cols + 1)) / cols
        cell_h = (self.max_height - self.min_spacing * (rows + 1)) / rows
        
        for i, room_type in enumerate(room_types):
            col = i % cols
            row = i // cols
            
            # Choose room size
            room_w, room_h = self._choose_room_size(room_type, cell_w * 0.8, cell_h * 0.8, rng)
            
            # Center in grid cell
            cell_center_x = self.min_spacing + col * (cell_w + self.min_spacing) + cell_w / 2
            cell_center_y = self.min_spacing + row * (cell_h + self.min_spacing) + cell_h / 2
            
            x = max(0, cell_center_x - room_w / 2)
            y = max(0, cell_center_y - room_h / 2)
            
            # Ensure within bounds
            x = min(x, self.max_width - room_w)
            y = min(y, self.max_height - room_h)
            
            rooms.append({
                "type": room_type,
                "position": {"x": int(x), "y": int(y)},
                "size": {"width": int(room_w), "length": int(room_h)}
            })
            
        return {"layout": {"rooms": rooms}}
    
    def _generate_l_shape(self, room_types: List[str], rng: random.Random) -> Dict:
        """Generate L-shaped layout."""
        rooms = []
        
        # Start with main room (usually living room) in corner
        main_room = room_types[0] if room_types else "Living Room"
        main_w, main_h = self._choose_room_size(main_room, 16, 16, rng)
        
        rooms.append({
            "type": main_room,
            "position": {"x": 0, "y": 0},
            "size": {"width": int(main_w), "length": int(main_h)}
        })
        
        # Place remaining rooms along the L-shape
        current_x = main_w + self.min_spacing
        current_y = 0
        
        for room_type in room_types[1:]:
            room_w, room_h = self._choose_room_size(room_type, 12, 12, rng)
            
            # Check if room fits horizontally
            if current_x + room_w <= self.max_width:
                x, y = current_x, current_y
                current_x += room_w + self.min_spacing
            else:
                # Move to vertical arm of L
                current_x = 0
                current_y = main_h + self.min_spacing
                x, y = current_x, current_y
                current_y += room_h + self.min_spacing
                
            # Ensure within bounds
            if x + room_w > self.max_width or y + room_h > self.max_height:
                break
                
            rooms.append({
                "type": room_type,
                "position": {"x": int(x), "y": int(y)},
                "size": {"width": int(room_w), "length": int(room_h)}
            })
            
        return {"layout": {"rooms": rooms}}
    
    def _generate_corridor_layout(self, room_types: List[str], rng: random.Random) -> Dict:
        """Generate layout with central corridor."""
        rooms = []
        
        # Central hallway
        hall_w = 4
        hall_h = min(32, self.max_height - 4)
        hall_x = (self.max_width - hall_w) // 2
        hall_y = 2
        
        rooms.append({
            "type": "Hallway",
            "position": {"x": int(hall_x), "y": int(hall_y)},
            "size": {"width": int(hall_w), "length": int(hall_h)}
        })
        
        # Place rooms on both sides of corridor
        left_x = 0
        right_x = hall_x + hall_w + self.min_spacing
        current_y = 2
        
        side = 0  # Alternate sides
        for room_type in room_types:
            if room_type == "Hallway":
                continue
                
            max_room_w = hall_x - self.min_spacing if side == 0 else self.max_width - right_x
            room_w, room_h = self._choose_room_size(room_type, max_room_w * 0.8, 14, rng)
            
            if side == 0:
                x = left_x
            else:
                x = right_x
                
            y = current_y
            
            # Check bounds
            if x + room_w > self.max_width or y + room_h > self.max_height:
                break
                
            rooms.append({
                "type": room_type,
                "position": {"x": int(x), "y": int(y)},
                "size": {"width": int(room_w), "length": int(room_h)}
            })
            
            if side == 0:
                current_y += room_h + self.min_spacing
            side = 1 - side  # Alternate sides
            
        return {"layout": {"rooms": rooms}}
    
    def _generate_compact_grid(self, room_types: List[str], rng: random.Random) -> Dict:
        """Generate very compact grid for many rooms.""" 
        rooms = []
        
        # Force 3x2 grid maximum
        cols, rows = 3, 2
        cell_w = (self.max_width - self.min_spacing * (cols + 1)) / cols
        cell_h = (self.max_height - self.min_spacing * (rows + 1)) / rows
        
        for i, room_type in enumerate(room_types[:6]):  # Limit to 6 rooms
            col = i % cols
            row = i // cols
            
            # Conservative room sizes
            max_w = min(12, cell_w * 0.7)
            max_h = min(12, cell_h * 0.7)
            room_w, room_h = self._choose_room_size(room_type, max_w, max_h, rng)
            
            # Position in grid
            cell_center_x = self.min_spacing + col * (cell_w + self.min_spacing) + cell_w / 2
            cell_center_y = self.min_spacing + row * (cell_h + self.min_spacing) + cell_h / 2
            
            x = max(0, cell_center_x - room_w / 2)
            y = max(0, cell_center_y - room_h / 2)
            
            rooms.append({
                "type": room_type,
                "position": {"x": int(x), "y": int(y)},
                "size": {"width": int(room_w), "length": int(room_h)}
            })
            
        return {"layout": {"rooms": rooms}}
    
    def _choose_room_size(self, room_type: str, max_w: float, max_h: float, 
                         rng: random.Random) -> Tuple[float, float]:
        """Choose appropriate size for room type within constraints."""
        size_options = REALISTIC_ROOM_SIZES.get(room_type, [(10, 10)])
        
        # Filter options that fit within constraints
        valid_options = [
            (w, h) for w, h in size_options 
            if w <= max_w and h <= max_h
        ]
        
        if not valid_options:
            # Scale down the smallest option
            w, h = min(size_options, key=lambda x: x[0] * x[1])
            scale = min(max_w / w, max_h / h, 1.0)
            return w * scale, h * scale
            
        return rng.choice(valid_options)


def generate_realistic_dataset(num_samples: int = 1000, 
                             output_path: str = "dataset/realistic_layouts.jsonl",
                             seed: int = None) -> None:
    """Generate dataset of realistic layouts."""
    if seed is not None:
        random.seed(seed)
        
    generator = RealisticLayoutGenerator()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            rng = random.Random(seed + i if seed else None)
            
            # Generate varied parameters
            params = {
                "houseStyle": rng.choice(["Modern", "Traditional", "Craftsman", "Ranch"]),
                "squareFeet": rng.choice([1200, 1500, 1800, 2200, 2500]),
                "bedrooms": rng.randint(2, 4),
                "bathrooms": {"full": rng.randint(1, 3)},
                "garage": {"attached": rng.choice([True, False])}
            }
            
            # Generate layout
            layout = generator.generate_layout(params, rng)
            
            # Create training sample
            sample = {
                "params": params,
                "layout": layout
            }
            
            f.write(json.dumps(sample) + '\n')
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} realistic layouts")
                
    logger.info(f"Generated {num_samples} realistic layouts to {output_path}")


if __name__ == "__main__":
    generate_realistic_dataset(1000, seed=42)
