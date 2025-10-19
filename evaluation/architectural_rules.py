"""
Architectural rules and validation for realistic house layouts.

Implements common-sense rules for room placement, garage access,
space utilization, and functional relationships.
"""
import logging
from typing import Dict, List, Tuple, Set
from collections import deque
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
    """Measure contiguous unused region using an iterative flood fill (avoids recursion limits)."""
    if (
        start_x < 0
        or start_x >= grid_w
        or start_y < 0
        or start_y >= grid_h
        or occupancy[start_y][start_x]
        or visited[start_y][start_x]
    ):
        return 0

    q = deque([(start_x, start_y)])
    visited[start_y][start_x] = True
    area = 0

    while q:
        x, y = q.popleft()
        area += 1
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < grid_w
                and 0 <= ny < grid_h
                and not occupancy[ny][nx]
                and not visited[ny][nx]
            ):
                visited[ny][nx] = True
                q.append((nx, ny))
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
    
    # NEW: Entry/exit validation
    all_issues.extend(validate_entry_access(layout, boundary_width, boundary_height))
    
    # NEW: Room size validation
    all_issues.extend(validate_room_sizes(layout))
    
    # NEW: Privacy rules
    all_issues.extend(validate_privacy_rules(layout))
    
    # NEW: Natural light and ventilation
    all_issues.extend(validate_natural_light(layout, boundary_width, boundary_height))
    
    # NEW: Circulation and traffic flow
    all_issues.extend(validate_traffic_flow(layout))
    
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
    
    # Fix 3: Only log room size issues (don't auto-resize to avoid overlaps)
    min_sizes = {
        "bedroom": (9, 9), "bathroom": (5, 6), "kitchen": (8, 8),
        "living room": (12, 12), "dining room": (8, 10), "garage": (12, 20),
        "laundry room": (5, 6), "office": (8, 8), "closet": (3, 4)
    }
    
    for room in rooms_to_keep:
        room_type = room.get("type", "").lower()
        size = room["size"]
        w, h = size["width"], size["length"]
        
        # Find matching room type and log if undersized (but don't fix to avoid overlaps)
        for room_key, (min_w, min_h) in min_sizes.items():
            if room_key in room_type:
                if w < min_w or h < min_h:
                    logger.warning(f"{room['type']} is undersized: {w}x{h} (recommended: {min_w}x{min_h})")
                break
    
    # Fix 4: Log living room boundary access (don't move to avoid overlaps)
    for room in rooms_to_keep:
        room_type = room.get("type", "").lower()
        if "living room" in room_type:
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
                logger.warning(f"Living room doesn't touch boundary - no clear main entrance")
    
    # Update layout
    fixed_layout = {
        "layout": {
            "rooms": rooms_to_keep,
            "dimensions": layout.get("layout", {}).get("dimensions", {})
        }
    }
    
    return fixed_layout


def validate_entry_access(layout: Dict, boundary_width: float, boundary_height: float) -> List[str]:
    """Validate that there's proper entry/exit access to the house."""
    issues = []
    rooms = layout.get("layout", {}).get("rooms", [])
    
    # Check for main entrance - living room or foyer should touch boundary
    entry_rooms = ["living room", "foyer", "entry", "entryway"]
    has_main_entry = False
    
    for room in rooms:
        room_type = room.get("type", "").lower()
        if any(entry in room_type for entry in entry_rooms):
            pos = room["position"]
            size = room["size"]
            x, y = pos["x"], pos["y"]
            w, h = size["width"], size["length"]
            
            # Check if room touches boundary (preferably front - bottom boundary)
            touches_front = y + h >= boundary_height - 1.0  # Front door
            touches_side = (x <= 1.0 or x + w >= boundary_width - 1.0)  # Side entry
            
            if touches_front or touches_side:
                has_main_entry = True
                break
    
    if not has_main_entry:
        issues.append("No clear main entrance - living room should touch exterior boundary")
    
    # Check that garage entry doesn't conflict with main entry
    garage_at_front = False
    main_entry_at_front = False
    
    for room in rooms:
        room_type = room.get("type", "").lower()
        pos = room["position"]
        size = room["size"]
        y = pos["y"]
        h = size["length"]
        
        if "garage" in room_type and y + h >= boundary_height - 1.0:
            garage_at_front = True
        elif any(entry in room_type for entry in entry_rooms) and y + h >= boundary_height - 1.0:
            main_entry_at_front = True
    
    if garage_at_front and main_entry_at_front:
        issues.append("Garage and main entrance both at front - may cause access conflicts")
    
    return issues


def validate_room_sizes(layout: Dict) -> List[str]:
    """Validate that room sizes are reasonable for their function."""
    issues = []
    rooms = layout.get("layout", {}).get("rooms", [])
    
    # Minimum reasonable sizes (width x length in feet)
    min_sizes = {
        "bedroom": (9, 9),     # 9x9 = 81 sq ft minimum
        "bathroom": (5, 6),    # 5x6 = 30 sq ft minimum  
        "kitchen": (8, 8),     # 8x8 = 64 sq ft minimum
        "living room": (12, 12), # 12x12 = 144 sq ft minimum
        "dining room": (8, 10), # 8x10 = 80 sq ft minimum
        "garage": (12, 20),    # 12x20 = 240 sq ft minimum (1 car)
        "laundry room": (5, 6), # 5x6 = 30 sq ft minimum
        "office": (8, 8),      # 8x8 = 64 sq ft minimum
        "closet": (3, 4),      # 3x4 = 12 sq ft minimum
    }
    
    # Maximum reasonable sizes  
    max_sizes = {
        "bedroom": (20, 20),   # 400 sq ft maximum
        "bathroom": (12, 15),  # 180 sq ft maximum
        "kitchen": (18, 20),   # 360 sq ft maximum 
        "living room": (25, 30), # 750 sq ft maximum
        "dining room": (16, 18), # 288 sq ft maximum
        "garage": (30, 30),    # 900 sq ft maximum (3+ cars)
        "laundry room": (10, 12), # 120 sq ft maximum
        "office": (15, 15),    # 225 sq ft maximum
        "closet": (8, 10),     # 80 sq ft maximum
    }
    
    for room in rooms:
        room_type = room.get("type", "").lower()
        size = room["size"]
        w, h = size["width"], size["length"]
        
        # Find matching room type
        matched_type = None
        for room_key in min_sizes.keys():
            if room_key in room_type:
                matched_type = room_key
                break
        
        if matched_type:
            min_w, min_h = min_sizes[matched_type]
            max_w, max_h = max_sizes[matched_type]
            
            # Check minimum size
            if w < min_w or h < min_h:
                issues.append(f"{room['type']} too small ({w}x{h}ft) - minimum should be {min_w}x{min_h}ft")
            
            # Check maximum size (less critical)
            if w > max_w or h > max_h:
                issues.append(f"{room['type']} unusually large ({w}x{h}ft) - typical maximum is {max_w}x{max_h}ft")
    
    return issues


def validate_privacy_rules(layout: Dict) -> List[str]:
    """Validate privacy considerations for bedrooms and bathrooms."""
    issues = []
    rooms = layout.get("layout", {}).get("rooms", [])
    
    # Get room positions
    bedrooms = []
    bathrooms = []
    public_rooms = []  # Living room, kitchen, dining room
    
    for room in rooms:
        room_type = room.get("type", "").lower()
        pos = room["position"]
        size = room["size"]
        center_x = pos["x"] + size["width"] / 2
        center_y = pos["y"] + size["length"] / 2
        
        if "bedroom" in room_type:
            bedrooms.append((center_x, center_y, room))
        elif "bathroom" in room_type:
            bathrooms.append((center_x, center_y, room))
        elif any(pub in room_type for pub in ["living", "kitchen", "dining"]):
            public_rooms.append((center_x, center_y, room))
    
    # Rule 1: Master bedroom should have some separation from public areas
    master_bedrooms = [r for r in bedrooms if "master" in r[2].get("type", "").lower()]
    for mx, my, master in master_bedrooms:
        for px, py, public in public_rooms:
            distance = math.sqrt((mx - px)**2 + (my - py)**2)
            if distance < 8.0:  # Less than 8 feet
                issues.append(f"Master bedroom too close to {public[2]['type']} - privacy concern")
    
    # Rule 2: Bathrooms shouldn't directly face main living areas
    for bx, by, bathroom in bathrooms:
        for px, py, public in public_rooms:
            # Check if bathroom door would face public room
            distance = math.sqrt((bx - px)**2 + (by - py)**2)
            if distance < 6.0:  # Very close
                issues.append(f"Bathroom very close to {public[2]['type']} - consider privacy screen")
    
    return issues


def validate_natural_light(layout: Dict, boundary_width: float, boundary_height: float) -> List[str]:
    """Validate that main rooms have access to natural light (exterior walls)."""
    issues = []
    rooms = layout.get("layout", {}).get("rooms", [])
    
    # Rooms that should have exterior walls for windows
    priority_rooms = ["living room", "kitchen", "bedroom", "dining room"]
    
    rooms_without_light = []
    
    for room in rooms:
        room_type = room.get("type", "").lower()
        if any(priority in room_type for priority in priority_rooms):
            pos = room["position"]
            size = room["size"]
            x, y = pos["x"], pos["y"]
            w, h = size["width"], size["length"]
            
            # Check if room touches any exterior boundary
            touches_exterior = (
                x <= 1.0 or  # Left wall
                y <= 1.0 or  # Top wall
                x + w >= boundary_width - 1.0 or  # Right wall
                y + h >= boundary_height - 1.0   # Bottom wall
            )
            
            if not touches_exterior:
                rooms_without_light.append(room["type"])
    
    if rooms_without_light:
        issues.append(f"Rooms lack natural light access: {', '.join(rooms_without_light)}")
    
    return issues


def validate_traffic_flow(layout: Dict) -> List[str]:
    """Validate logical traffic flow and circulation patterns."""
    issues = []
    rooms = layout.get("layout", {}).get("rooms", [])
    
    # Get room centers and types
    room_data = []
    for room in rooms:
        room_type = room.get("type", "").lower()
        pos = room["position"]
        size = room["size"]
        center_x = pos["x"] + size["width"] / 2
        center_y = pos["y"] + size["length"] / 2
        room_data.append((center_x, center_y, room_type, room))
    
    # Rule 1: Kitchen should be accessible from living/dining areas
    kitchens = [(x, y, t, r) for x, y, t, r in room_data if "kitchen" in t]
    social_rooms = [(x, y, t, r) for x, y, t, r in room_data if any(s in t for s in ["living", "dining"])]
    
    for kx, ky, kt, kr in kitchens:
        accessible = False
        for sx, sy, st, sr in social_rooms:
            distance = math.sqrt((kx - sx)**2 + (ky - sy)**2)
            if distance < 20.0:  # Within 20 feet
                accessible = True
                break
        
        if not accessible:
            issues.append("Kitchen isolated from living/dining areas - poor traffic flow")
    
    # Rule 2: No room should be completely isolated
    for i, (x1, y1, t1, r1) in enumerate(room_data):
        if "garage" in t1:  # Garages can be isolated
            continue
            
        min_distance = float('inf')
        for j, (x2, y2, t2, r2) in enumerate(room_data):
            if i != j and "garage" not in t2:
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                min_distance = min(min_distance, distance)
        
        if min_distance > 25.0:  # More than 25 feet from any other room
            issues.append(f"{r1['type']} is too isolated from other rooms")
    
    # Rule 3: Bedrooms shouldn't require passing through other bedrooms
    bedrooms = [(x, y, t, r) for x, y, t, r in room_data if "bedroom" in t]
    if len(bedrooms) > 1:
        for bx, by, bt, br in bedrooms:
            # Check if bedroom is surrounded by other bedrooms
            nearby_bedrooms = 0
            nearby_other = 0
            
            for ox, oy, ot, or_ in room_data:
                if br != or_:  # Different room
                    distance = math.sqrt((bx - ox)**2 + (by - oy)**2)
                    if distance < 15.0:  # Nearby room
                        if "bedroom" in ot:
                            nearby_bedrooms += 1
                        else:
                            nearby_other += 1
            
            if nearby_bedrooms > 1 and nearby_other == 0:
                issues.append(f"{br['type']} may require passing through other bedrooms to access")
    
    return issues
