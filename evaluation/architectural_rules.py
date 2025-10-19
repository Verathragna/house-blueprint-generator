"""
Architectural rules and validation for realistic house layouts.

Implements common-sense rules for room placement, garage access,
space utilization, and functional relationships.
"""
import logging
from typing import Dict, List, Tuple, Set
import math

logger = logging.getLogger(__name__)

def validate_garage_access(layout: Dict, boundary_width: float, boundary_height: float) -> List[str]:
    """Ensure garage has vehicle access (touches exterior boundary)."""
    issues = []
    rooms = layout.get("layout", {}).get("rooms", [])
    
    for room in rooms:
        if "garage" in room.get("type", "").lower():
            pos = room["position"]
            size = room["size"] 
            x, y = pos["x"], pos["y"]
            w, h = size["width"], size["length"]
            
            # Check if garage touches any boundary
            touches_boundary = (
                x <= 1.0 or  # Left boundary
                y <= 1.0 or  # Top boundary  
                x + w >= boundary_width - 1.0 or  # Right boundary
                y + h >= boundary_height - 1.0   # Bottom boundary
            )
            
            if not touches_boundary:
                issues.append(f"Garage must touch exterior boundary for vehicle access")
                logger.warning(f"Garage at ({x}, {y}) {w}x{h} doesn't touch boundary")
    
    return issues

def detect_unused_spaces(layout: Dict, boundary_width: float, boundary_height: float, 
                        min_space_threshold: float = 16.0) -> List[str]:
    """Detect unused rectangular spaces that could be better utilized."""
    issues = []
    rooms = layout.get("layout", {}).get("rooms", [])
    
    if not rooms:
        return issues
    
    # Create occupancy grid for analysis
    grid_res = 1.0  # 1 ft resolution
    grid_w = int(boundary_width / grid_res) + 1
    grid_h = int(boundary_height / grid_res) + 1
    occupancy = [[False] * grid_w for _ in range(grid_h)]
    
    # Mark occupied spaces
    for room in rooms:
        pos = room["position"]
        size = room["size"]
        x1 = max(0, int(pos["x"] / grid_res))
        y1 = max(0, int(pos["y"] / grid_res))
        x2 = min(grid_w, int((pos["x"] + size["width"]) / grid_res))
        y2 = min(grid_h, int((pos["y"] + size["length"]) / grid_res))
        
        for y in range(y1, y2):
            for x in range(x1, x2):
                if 0 <= y < grid_h and 0 <= x < grid_w:
                    occupancy[y][x] = True
    
    # Find unused rectangular regions
    unused_regions = []
    visited = [[False] * grid_w for _ in range(grid_h)]
    
    for y in range(grid_h):
        for x in range(grid_w):
            if not occupancy[y][x] and not visited[y][x]:
                # Found unoccupied space, measure its extent
                region_area = _measure_unused_region(occupancy, visited, x, y, grid_w, grid_h)
                region_area_sqft = region_area * (grid_res ** 2)
                
                if region_area_sqft >= min_space_threshold:
                    unused_regions.append(region_area_sqft)
                    issues.append(f"Unused space of {region_area_sqft:.0f} sq ft detected")
    
    if unused_regions:
        total_unused = sum(unused_regions)
        logger.warning(f"Total unused space: {total_unused:.0f} sq ft in {len(unused_regions)} regions")
    
    return issues

def _measure_unused_region(occupancy: List[List[bool]], visited: List[List[bool]], 
                          start_x: int, start_y: int, grid_w: int, grid_h: int) -> int:
    """Measure contiguous unused region using flood fill."""
    if (start_x < 0 or start_x >= grid_w or start_y < 0 or start_y >= grid_h or 
        occupancy[start_y][start_x] or visited[start_y][start_x]):
        return 0
    
    visited[start_y][start_x] = True
    area = 1
    
    # Check 4-connected neighbors
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        area += _measure_unused_region(occupancy, visited, start_x + dx, start_y + dy, grid_w, grid_h)
    
    return area

def validate_room_adjacencies(layout: Dict) -> List[str]:
    """Validate logical room adjacencies (kitchen-dining, bedroom-bathroom, etc.)."""
    issues = []
    rooms = layout.get("layout", {}).get("rooms", [])
    
    # Build room type lists
    room_positions = {}
    for room in rooms:
        room_type = room.get("type", "").lower()
        pos = room["position"]
        size = room["size"]
        center_x = pos["x"] + size["width"] / 2
        center_y = pos["y"] + size["length"] / 2
        
        if room_type not in room_positions:
            room_positions[room_type] = []
        room_positions[room_type].append((center_x, center_y, room))
    
    # Rule 1: Dining room should be adjacent to kitchen
    if "dining room" in room_positions and "kitchen" in room_positions:
        dining_rooms = room_positions["dining room"]
        kitchens = room_positions["kitchen"]
        
        for dx, dy, dining in dining_rooms:
            min_distance = float('inf')
            for kx, ky, kitchen in kitchens:
                distance = math.sqrt((dx - kx)**2 + (dy - ky)**2)
                min_distance = min(min_distance, distance)
            
            if min_distance > 15.0:  # More than 15 ft apart
                issues.append("Dining room should be adjacent to kitchen")
                logger.warning(f"Dining room {min_distance:.1f}ft from kitchen")
    
    # Rule 2: Master bathroom should be close to master bedroom
    master_bedrooms = [r for r in room_positions.get("bedroom", []) if "master" in r[2].get("type", "").lower()]
    master_bathrooms = [r for r in room_positions.get("bathroom", []) if "master" in r[2].get("type", "").lower()]
    
    if master_bedrooms and master_bathrooms:
        for bx, by, bedroom in master_bedrooms:
            min_distance = float('inf')
            for mx, my, bathroom in master_bathrooms:
                distance = math.sqrt((bx - mx)**2 + (by - my)**2)
                min_distance = min(min_distance, distance)
            
            if min_distance > 12.0:  # More than 12 ft apart
                issues.append("Master bathroom should be adjacent to master bedroom")
    
    # Rule 3: At least one bathroom should be accessible from bedrooms
    if "bedroom" in room_positions and "bathroom" in room_positions:
        bedrooms = room_positions["bedroom"]
        bathrooms = room_positions["bathroom"]
        
        accessible_bathrooms = 0
        for bx, by, bedroom in bedrooms:
            for rx, ry, bathroom in bathrooms:
                distance = math.sqrt((bx - rx)**2 + (by - ry)**2)
                if distance <= 20.0:  # Within 20 ft
                    accessible_bathrooms += 1
                    break
        
        if accessible_bathrooms == 0:
            issues.append("No bathroom accessible from bedrooms")
    
    return issues

def detect_useless_hallways(layout: Dict) -> List[str]:
    """Detect hallways that don't serve a functional purpose."""
    issues = []
    rooms = layout.get("layout", {}).get("rooms", [])
    
    hallways = [r for r in rooms if "hallway" in r.get("type", "").lower()]
    other_rooms = [r for r in rooms if "hallway" not in r.get("type", "").lower()]
    
    for hallway in hallways:
        h_pos = hallway["position"] 
        h_size = hallway["size"]
        h_x1, h_y1 = h_pos["x"], h_pos["y"]
        h_x2, h_y2 = h_x1 + h_size["width"], h_y1 + h_size["length"]
        
        # Check if hallway actually connects rooms (not isolated)
        connected_rooms = 0
        for room in other_rooms:
            r_pos = room["position"]
            r_size = room["size"]  
            r_x1, r_y1 = r_pos["x"], r_pos["y"]
            r_x2, r_y2 = r_x1 + r_size["width"], r_y1 + r_size["length"]
            
            # Check if hallway and room are adjacent (shared boundary)
            adjacent = (
                (abs(h_x2 - r_x1) < 1.0 and not (h_y2 <= r_y1 or h_y1 >= r_y2)) or  # Right edge
                (abs(h_x1 - r_x2) < 1.0 and not (h_y2 <= r_y1 or h_y1 >= r_y2)) or  # Left edge  
                (abs(h_y2 - r_y1) < 1.0 and not (h_x2 <= r_x1 or h_x1 >= r_x2)) or  # Bottom edge
                (abs(h_y1 - r_y2) < 1.0 and not (h_x2 <= r_x1 or h_x1 >= r_x2))     # Top edge
            )
            
            if adjacent:
                connected_rooms += 1
        
        # Hallway should connect at least 2 rooms to be useful
        if connected_rooms < 2:
            issues.append(f"Hallway connects only {connected_rooms} rooms - likely unnecessary")
            logger.warning(f"Useless hallway at ({h_x1}, {h_y1}) {h_size['width']}x{h_size['length']}")
        
        # Check for very small hallways that are likely artifacts
        hallway_area = h_size["width"] * h_size["length"]
        if hallway_area < 20.0:  # Less than 20 sq ft
            issues.append(f"Hallway too small ({hallway_area:.0f} sq ft) - likely unnecessary")
    
    return issues

def validate_architectural_rules(layout: Dict, boundary_width: float = 40.0, 
                                boundary_height: float = 40.0) -> List[str]:
    """Run all architectural rule validations."""
    all_issues = []
    
    # Garage access validation
    all_issues.extend(validate_garage_access(layout, boundary_width, boundary_height))
    
    # Unused space detection  
    all_issues.extend(detect_unused_spaces(layout, boundary_width, boundary_height))
    
    # Room adjacency rules
    all_issues.extend(validate_room_adjacencies(layout))
    
    # Useless hallway detection
    all_issues.extend(detect_useless_hallways(layout))
    
    if all_issues:
        logger.info(f"Architectural validation found {len(all_issues)} issues")
    
    return all_issues

def fix_architectural_issues(layout: Dict, boundary_width: float = 40.0, 
                            boundary_height: float = 40.0) -> Dict:
    """Attempt to fix common architectural issues automatically."""
    rooms = layout.get("layout", {}).get("rooms", [])
    if not rooms:
        return layout
    
    # Fix 1: Move garage to boundary if not already there
    for room in rooms:
        if "garage" in room.get("type", "").lower():
            pos = room["position"]
            size = room["size"]
            x, y = pos["x"], pos["y"]
            w, h = size["width"], size["length"]
            
            # Check if already touches boundary
            touches_boundary = (
                x <= 1.0 or y <= 1.0 or 
                x + w >= boundary_width - 1.0 or 
                y + h >= boundary_height - 1.0
            )
            
            if not touches_boundary:
                # Move garage to right boundary (common for attached garages)
                new_x = boundary_width - w
                room["position"]["x"] = new_x
                logger.info(f"Moved garage to boundary: ({new_x}, {y})")
    
    # Fix 2: Remove tiny hallways
    rooms_to_keep = []
    for room in rooms:
        if "hallway" in room.get("type", "").lower():
            area = room["size"]["width"] * room["size"]["length"]
            if area >= 20.0:  # Keep hallways >= 20 sq ft
                rooms_to_keep.append(room)
            else:
                logger.info(f"Removed tiny hallway: {area:.0f} sq ft")
        else:
            rooms_to_keep.append(room)
    
    # Update layout
    fixed_layout = {
        "layout": {
            "rooms": rooms_to_keep,
            "dimensions": layout.get("layout", {}).get("dimensions", {})
        }
    }
    
    return fixed_layout