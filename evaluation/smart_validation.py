"""
Smart validation system that prevents infinite loops and gracefully degrades constraints.

Key improvements:
1. Early detection of impossible layouts
2. Progressive constraint relaxation  
3. Faster termination conditions
4. Smarter room placement algorithms
"""
import logging
from typing import Dict, List, Tuple, Optional
import math

logger = logging.getLogger(__name__)

def assess_layout_feasibility(layout: Dict, max_width: float, max_height: float) -> Dict:
    """
    Quickly assess if a layout is fundamentally feasible before expensive processing.
    
    Returns assessment with recommended actions.
    """
    rooms = layout.get("layout", {}).get("rooms", [])
    if not rooms:
        return {"feasible": True, "issues": [], "recommended_action": "proceed"}
    
    assessment = {
        "feasible": True,
        "issues": [],
        "recommended_action": "proceed",
        "severity": "low"
    }
    
    # 1. Calculate total room area vs available space
    total_room_area = sum(
        float(room.get("size", {}).get("width", 0)) * 
        float(room.get("size", {}).get("length", 0))
        for room in rooms
    )
    available_area = max_width * max_height
    density_ratio = total_room_area / available_area if available_area > 0 else 1.0
    
    # 2. Count severe overlaps (rooms completely inside others)
    severe_overlaps = count_severe_overlaps(rooms)
    
    # 3. Count boundary violations  
    boundary_violations = count_boundary_violations(rooms, max_width, max_height)
    
    # 4. Assess room count vs space
    room_count = len(rooms)
    area_per_room = available_area / room_count if room_count > 0 else 0
    
    # Decision matrix
    if density_ratio > 0.8:
        assessment["issues"].append(f"Excessive density: {density_ratio:.1%} of space used")
        assessment["severity"] = "high"
        
    if severe_overlaps > room_count * 0.3:  # More than 30% severe overlaps
        assessment["issues"].append(f"Too many severe overlaps: {severe_overlaps}/{room_count}")
        assessment["severity"] = "high"
        
    if boundary_violations > room_count * 0.4:  # More than 40% outside bounds
        assessment["issues"].append(f"Many boundary violations: {boundary_violations}/{room_count}")
        assessment["severity"] = "high"
        
    if area_per_room < 50:  # Less than 50 sq ft per room on average
        assessment["issues"].append(f"Insufficient space per room: {area_per_room:.0f} sq ft avg")
        assessment["severity"] = "high"
        
    # Determine recommended action
    if assessment["severity"] == "high":
        assessment["feasible"] = False
        if density_ratio > 0.9 or severe_overlaps > room_count * 0.5:
            assessment["recommended_action"] = "emergency_simplify"
        else:
            assessment["recommended_action"] = "aggressive_shrink"
    elif assessment["severity"] == "medium" or len(assessment["issues"]) > 2:
        assessment["recommended_action"] = "moderate_shrink"
        
    logger.info(f"Layout feasibility: {assessment['severity']} severity, "
               f"density={density_ratio:.1%}, overlaps={severe_overlaps}, "
               f"boundary_violations={boundary_violations}")
    
    return assessment

def count_severe_overlaps(rooms: List[Dict]) -> int:
    """Count rooms that are mostly or completely overlapping with others."""
    severe_overlaps = 0
    
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            room1, room2 = rooms[i], rooms[j]
            overlap_ratio = calculate_overlap_ratio(room1, room2)
            
            if overlap_ratio > 0.5:  # More than 50% overlap
                severe_overlaps += 1
                
    return severe_overlaps

def count_boundary_violations(rooms: List[Dict], max_width: float, max_height: float) -> int:
    """Count rooms that significantly violate boundaries."""
    violations = 0
    
    for room in rooms:
        x = float(room.get("position", {}).get("x", 0))
        y = float(room.get("position", {}).get("y", 0)) 
        w = float(room.get("size", {}).get("width", 0))
        h = float(room.get("size", {}).get("length", 0))
        
        # Check for significant violations (not just minor boundary issues)
        if (x < -5 or y < -5 or  # Significantly outside bounds
            x + w > max_width + 5 or y + h > max_height + 5 or
            w <= 0 or h <= 0):  # Invalid dimensions
            violations += 1
            
    return violations

def calculate_overlap_ratio(room1: Dict, room2: Dict) -> float:
    """Calculate what fraction of the smaller room overlaps with the larger."""
    # Get room bounds
    x1 = float(room1.get("position", {}).get("x", 0))
    y1 = float(room1.get("position", {}).get("y", 0))
    w1 = float(room1.get("size", {}).get("width", 0))
    h1 = float(room1.get("size", {}).get("length", 0))
    
    x2 = float(room2.get("position", {}).get("x", 0))
    y2 = float(room2.get("position", {}).get("y", 0))
    w2 = float(room2.get("size", {}).get("width", 0))
    h2 = float(room2.get("size", {}).get("length", 0))
    
    # Calculate overlap area
    if (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2):
        overlap_w = min(x1 + w1, x2 + w2) - max(x1, x2)
        overlap_h = min(y1 + h1, y2 + h2) - max(y1, y2)
        overlap_area = max(0, overlap_w) * max(0, overlap_h)
        
        # Calculate ratio relative to smaller room
        area1 = w1 * h1
        area2 = w2 * h2
        smaller_area = min(area1, area2)
        
        return overlap_area / smaller_area if smaller_area > 0 else 0
    
    return 0

def progressive_constraint_relaxation(layout: Dict, max_width: float, max_height: float,
                                    min_separation: float = 0.5, max_iterations: int = 50) -> Dict:
    """
    Apply constraints with progressive relaxation to avoid infinite loops.
    
    Instead of trying to solve everything at once, relax constraints gradually.
    """
    from evaluation.validators import validate_layout, enforce_min_separation, clamp_bounds
    
    current_layout = layout
    separation_targets = [min_separation, min_separation * 0.7, min_separation * 0.5, 0.2, 0.0]
    iteration_limits = [20, 15, 10, 8, 5]
    
    logger.info("Starting progressive constraint relaxation...")
    
    for stage, (target_sep, max_iters) in enumerate(zip(separation_targets, iteration_limits)):
        logger.info(f"Stage {stage + 1}: target_separation={target_sep:.1f}, max_iterations={max_iters}")
        
        # Apply separation with limited iterations
        if target_sep > 0:
            current_layout = enforce_min_separation(
                current_layout,
                target_sep,
                max_width=max_width,
                max_length=max_height,
                max_iterations=max_iters
            )
        
        # Clamp to bounds
        current_layout = clamp_bounds(current_layout, max_width, max_height)
        
        # Check if we've achieved acceptable quality
        issues = validate_layout(
            current_layout,
            max_width=max_width,
            max_length=max_height,
            min_separation=0.1,  # Very lenient for checking
            adjacency=None,
            require_connectivity=False
        )
        
        # Count only serious issues (overlaps, bounds violations)
        serious_issues = [issue for issue in issues if 
                         "overlap" in issue.lower() or "boundary" in issue.lower()]
        
        if len(serious_issues) <= 2:  # Acceptable quality
            logger.info(f"Progressive relaxation succeeded at stage {stage + 1}")
            break
            
        if stage == len(separation_targets) - 1:  # Final stage
            logger.warning("Progressive relaxation completed with remaining issues")
    
    return current_layout

def smart_room_placement(layout: Dict, max_width: float, max_height: float) -> Dict:
    """
    Intelligently reposition rooms to minimize conflicts using grid-based approach.
    
    Much faster than iterative separation algorithms.
    """
    rooms = layout.get("layout", {}).get("rooms", [])
    if not rooms:
        return layout
    
    logger.info(f"Smart placement for {len(rooms)} rooms in {max_width}x{max_height} space")
    
    # Calculate optimal grid
    n_rooms = len(rooms)
    cols = math.ceil(math.sqrt(n_rooms * max_width / max_height))
    rows = math.ceil(n_rooms / cols)
    
    cell_w = max_width / cols
    cell_h = max_height / rows
    
    # Sort rooms by area (place larger rooms first)
    rooms_with_area = []
    for room in rooms:
        w = float(room.get("size", {}).get("width", 0))
        h = float(room.get("size", {}).get("length", 0))
        area = w * h
        rooms_with_area.append((area, room))
    
    rooms_with_area.sort(key=lambda x: x[0], reverse=True)  # Sort by area, largest first
    
    # Place rooms in grid
    placed_rooms = []
    for i, (area, room) in enumerate(rooms_with_area):
        col = i % cols
        row = i // cols
        
        # Calculate cell center
        cell_center_x = (col + 0.5) * cell_w
        cell_center_y = (row + 0.5) * cell_h
        
        # Get room dimensions
        room_w = float(room.get("size", {}).get("width", 0))
        room_h = float(room.get("size", {}).get("length", 0))
        
        # Scale room to fit in cell with guaranteed spacing
        max_cell_w = cell_w * 0.8  # Leave 20% margin
        max_cell_h = cell_h * 0.8
        
        if room_w > max_cell_w or room_h > max_cell_h:
            # Scale to fit the most constraining dimension
            scale_w = max_cell_w / room_w if room_w > 0 else 1.0
            scale_h = max_cell_h / room_h if room_h > 0 else 1.0
            scale = min(scale_w, scale_h, 1.0)  # Don't make rooms bigger
            
            room_w = room_w * scale
            room_h = room_h * scale
            room["size"]["width"] = int(room_w)
            room["size"]["length"] = int(room_h)
        
        # Position room at cell center with bounds checking
        x = cell_center_x - room_w/2
        y = cell_center_y - room_h/2
        
        # Ensure room stays within its designated cell and overall bounds
        cell_left = col * cell_w
        cell_top = row * cell_h
        cell_right = (col + 1) * cell_w
        cell_bottom = (row + 1) * cell_h
        
        x = max(cell_left, min(x, cell_right - room_w))
        y = max(cell_top, min(y, cell_bottom - room_h))
        
        # Final boundary check
        x = max(0, min(x, max_width - room_w))
        y = max(0, min(y, max_height - room_h))
        
        room["position"]["x"] = int(x)
        room["position"]["y"] = int(y)
        placed_rooms.append(room)
    
    # Update layout
    new_layout = {
        "layout": {
            "rooms": placed_rooms,
            "dimensions": layout.get("layout", {}).get("dimensions", {})
        }
    }
    
    logger.info("Smart room placement completed")
    return new_layout

def guaranteed_non_overlapping_layout(layout: Dict, max_width: float, max_height: float) -> Dict:
    """
    Create a layout that is guaranteed to have no overlaps by using minimal room sizes
    and strict grid placement. This is the ultimate fallback.
    """
    rooms = layout.get("layout", {}).get("rooms", [])
    if not rooms:
        return layout
    
    logger.info(f"Creating guaranteed non-overlapping layout for {len(rooms)} rooms")
    
    # Limit rooms if too many
    if len(rooms) > 8:
        # Keep most important rooms
        room_priorities = {
            "living room": 1, "bedroom": 2, "kitchen": 3, "bathroom": 4,
            "dining room": 5, "garage": 6, "laundry room": 7, "office": 8
        }
        rooms_with_priority = []
        for room in rooms:
            room_type = room.get("type", "").lower()
            priority = 99
            for key, prio in room_priorities.items():
                if key in room_type:
                    priority = prio
                    break
            rooms_with_priority.append((priority, room))
        
        rooms_with_priority.sort()  # Sort by priority
        rooms = [room for _, room in rooms_with_priority[:8]]
        logger.info(f"Limited to {len(rooms)} priority rooms")
    
    # Calculate grid that definitely fits
    n_rooms = len(rooms)
    cols = min(4, math.ceil(math.sqrt(n_rooms)))
    rows = math.ceil(n_rooms / cols)
    
    # Calculate cell size with guaranteed margins
    cell_w = max_width / cols
    cell_h = max_height / rows
    
    # Set maximum room size to ensure no overlaps
    max_room_w = max(6, cell_w * 0.7)  # At least 6ft wide
    max_room_h = max(6, cell_h * 0.7)  # At least 6ft long
    
    logger.info(f"Grid: {cols}x{rows}, cell size: {cell_w:.1f}x{cell_h:.1f}, max room: {max_room_w:.1f}x{max_room_h:.1f}")
    
    # Place each room in its designated grid cell
    final_rooms = []
    for i, room in enumerate(rooms):
        col = i % cols
        row = i // cols
        
        # Set room size to fit perfectly in cell
        room_w = min(max_room_w, cell_w - 2)  # 2ft margin
        room_h = min(max_room_h, cell_h - 2)  # 2ft margin
        
        # Position in cell with margin
        x = col * cell_w + 1  # 1ft margin from cell edge
        y = row * cell_h + 1  # 1ft margin from cell edge
        
        # Update room
        room["size"]["width"] = int(room_w)
        room["size"]["length"] = int(room_h)
        room["position"]["x"] = int(x)
        room["position"]["y"] = int(y)
        
        final_rooms.append(room)
        
        logger.info(f"Placed {room['type']} at ({int(x)}, {int(y)}) size {int(room_w)}x{int(room_h)}")
    
    return {
        "layout": {
            "rooms": final_rooms,
            "dimensions": layout.get("layout", {}).get("dimensions", {})
        }
    }
