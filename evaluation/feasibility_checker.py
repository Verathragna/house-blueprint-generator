"""
Feasibility checker for house layout requests.

Validates that requested rooms can physically fit in the given space
before attempting generation. Provides clear feedback to users.
"""
import math
from typing import Dict, List, Tuple, Optional

# Minimum room sizes (width x length in feet)
MIN_ROOM_SIZES = {
    "living room": (14, 14),    # 196 sq ft
    "kitchen": (10, 12),        # 120 sq ft  
    "bedroom": (10, 10),        # 100 sq ft
    "bathroom": (6, 8),         # 48 sq ft
    "dining room": (10, 10),    # 100 sq ft
    "garage": (12, 20),         # 240 sq ft (1 car)
    "laundry room": (6, 8),     # 48 sq ft
    "office": (8, 10),          # 80 sq ft
    "closet": (3, 4),           # 12 sq ft
    "hallway": (4, 8),          # 32 sq ft
}

# Space efficiency factors
CIRCULATION_FACTOR = 0.15      # 15% space needed for hallways/circulation
SPACING_FACTOR = 0.10          # 10% space needed for walls/spacing between rooms
USABLE_SPACE_RATIO = 1.0 - CIRCULATION_FACTOR - SPACING_FACTOR  # 75% usable
# Require some slack beyond bare minimum to allow walls, doors, and separation.
# If required area exceeds this ratio of usable space, treat as not feasible.
SAFETY_MARGIN_RATIO = 0.90  # require at least 10% headroom


def check_layout_feasibility(params: Dict) -> Tuple[bool, str, Dict]:
    """
    Check if the requested layout is physically feasible.
    
    Returns:
        (is_feasible, message, analysis)
    """
    # Get dimensions
    dimensions = params.get("dimensions", {})
    width = float(dimensions.get("width", 40))
    height = float(dimensions.get("depth", dimensions.get("height", 40)))
    total_area = width * height
    usable_area = total_area * USABLE_SPACE_RATIO
    
    # Calculate required rooms
    required_rooms = calculate_required_rooms(params)
    
    # Calculate minimum space needed
    min_area_needed = calculate_minimum_area(required_rooms)
    
    # Check basic feasibility
    if min_area_needed > usable_area:
        shortage = min_area_needed - usable_area
        message = (
            f"❌ Layout NOT FEASIBLE\n"
            f"Space needed: {min_area_needed:.0f} sq ft\n"
            f"Space available: {usable_area:.0f} sq ft (75% of {total_area:.0f} sq ft)\n"
            f"Shortage: {shortage:.0f} sq ft\n\n"
            f"💡 Suggestions:\n"
            f"• Increase dimensions to {width + math.ceil(shortage / (USABLE_SPACE_RATIO * height))}x{height} ft or {width}x{height + math.ceil(shortage / (USABLE_SPACE_RATIO * width))} ft\n"
            f"• Reduce bedrooms by {max(1, len([r for r in required_rooms if 'bedroom' in r.lower()]) - 2)}\n"
            f"• Remove garage (-240 sq ft) or laundry room (-48 sq ft)"
        )
        
        analysis = {
            "feasible": False,
            "total_area": total_area,
            "usable_area": usable_area,
            "required_area": min_area_needed,
            "shortage": shortage,
            "room_breakdown": get_room_breakdown(required_rooms),
            "suggestions": generate_suggestions(params, shortage, width, height)
        }
        
        return False, message, analysis

    # Require safety margin for generation robustness
    margin_cap = usable_area * SAFETY_MARGIN_RATIO
    if min_area_needed > margin_cap:
        shortage = min_area_needed - margin_cap
        efficiency = (min_area_needed / usable_area) * 100
        message = (
            f"❌ Layout NOT FEASIBLE - Too tight ({efficiency:.1f}% of usable)\n"
            f"Minimum room area leaves insufficient slack for walls, doors and separation.\n"
            f"Space available (usable): {usable_area:.0f} sq ft; Required minimum: {min_area_needed:.0f} sq ft;\n"
            f"Need at least {(1-SAFETY_MARGIN_RATIO)*100:.0f}% headroom.\n\n"
            f"💡 Suggestions:\n"
            f"• Increase dimensions to {width + math.ceil(shortage / (USABLE_SPACE_RATIO * height))}x{height} ft or {width}x{height + math.ceil(shortage / (USABLE_SPACE_RATIO * width))} ft\n"
            f"• Or reduce room counts (e.g., bedrooms/bathrooms) or remove garage"
        )
        analysis = {
            "feasible": False,
            "total_area": total_area,
            "usable_area": usable_area,
            "required_area": min_area_needed,
            "shortage": shortage,
            "room_breakdown": get_room_breakdown(required_rooms),
            "suggestions": generate_suggestions(params, shortage, width, height),
        }
        return False, message, analysis
    
    # Check dimensional constraints (can rooms actually fit?)
    dimensional_issues = check_dimensional_constraints(required_rooms, width, height)
    
    if dimensional_issues:
        message = (
            f"❌ Layout NOT FEASIBLE - Dimensional Issues\n"
            f"Total area: {total_area:.0f} sq ft (sufficient)\n"
            f"Usable area: {usable_area:.0f} sq ft\n"
            f"Required area: {min_area_needed:.0f} sq ft\n\n"
            f"🚨 Issues:\n" + 
            "\n".join(f"• {issue}" for issue in dimensional_issues) +
            f"\n\n💡 Suggestion: Increase dimensions to at least 35x35 ft for complex layouts"
        )
        
        analysis = {
            "feasible": False,
            "total_area": total_area,
            "usable_area": usable_area,
            "required_area": min_area_needed,
            "dimensional_issues": dimensional_issues,
            "room_breakdown": get_room_breakdown(required_rooms)
        }
        
        return False, message, analysis
    
    # Layout is feasible
    efficiency = (min_area_needed / usable_area) * 100
    
    if efficiency > 90:
        feasibility_level = "Tight fit"
    elif efficiency > 75:
        feasibility_level = "Good fit"
    else:
        feasibility_level = "Spacious"
        
    message = (
        f"✅ Layout FEASIBLE - {feasibility_level}\n"
        f"Total area: {total_area:.0f} sq ft\n"
        f"Usable area: {usable_area:.0f} sq ft\n"
        f"Required area: {min_area_needed:.0f} sq ft\n"
        f"Space efficiency: {efficiency:.1f}%\n"
        f"Rooms: {len(required_rooms)} total"
    )
    
    analysis = {
        "feasible": True,
        "total_area": total_area,
        "usable_area": usable_area,
        "required_area": min_area_needed,
        "efficiency": efficiency,
        "feasibility_level": feasibility_level,
        "room_count": len(required_rooms),
        "room_breakdown": get_room_breakdown(required_rooms)
    }
    
    return True, message, analysis


def calculate_required_rooms(params: Dict) -> List[str]:
    """Calculate list of required rooms from parameters."""
    rooms = []
    
    # Essential rooms
    rooms.append("living room")
    rooms.append("kitchen")
    
    # Bedrooms
    bedrooms = int(params.get("bedrooms", 2))
    for i in range(bedrooms):
        if i == 0 and bedrooms >= 3:
            rooms.append("master bedroom")
        else:
            rooms.append("bedroom")
    
    # Bathrooms
    bathrooms = params.get("bathrooms", {})
    full_baths = int(bathrooms.get("full", 1))
    half_baths = int(bathrooms.get("half", 0))
    
    for i in range(full_baths):
        if i == 0 and bedrooms >= 3:
            rooms.append("master bathroom")
        else:
            rooms.append("bathroom")
    
    for i in range(half_baths):
        rooms.append("half bathroom")
    
    # Optional rooms based on size/parameters
    square_feet = int(params.get("squareFeet", 1500))
    
    if square_feet >= 1800 or params.get("diningRoom", True):
        rooms.append("dining room")
        
    if square_feet >= 2000 or params.get("laundryRoom", True):
        rooms.append("laundry room")
        
    if params.get("office", False):
        rooms.append("office")
        
    # Garage
    garage = params.get("garage", {})
    if garage.get("attached", False):
        cars = int(garage.get("cars", 1))
        if cars >= 2:
            rooms.append("2-car garage")
        else:
            rooms.append("garage")
    
    return rooms


def calculate_minimum_area(rooms: List[str]) -> float:
    """Calculate minimum area needed for all rooms."""
    total_area = 0
    
    for room in rooms:
        room_lower = room.lower()
        
        # Find matching room type
        min_area = 100  # Default 100 sq ft
        for room_type, (w, h) in MIN_ROOM_SIZES.items():
            if room_type in room_lower:
                min_area = w * h
                break
        
        # Special cases
        if "master" in room_lower:
            min_area = min_area * 1.5  # Master rooms are bigger
        elif "half bathroom" in room_lower:
            min_area = 24  # Half bath is smaller
        elif "2-car garage" in room_lower:
            min_area = 400  # 2-car garage
        elif "closet" in room_lower:
            min_area = 12  # Small closet
            
        total_area += min_area
        
    return total_area


def check_dimensional_constraints(rooms: List[str], width: float, height: float) -> List[str]:
    """Check if rooms can physically fit in the dimensions."""
    issues = []
    
    # Check for very large rooms that won't fit
    for room in rooms:
        room_lower = room.lower()
        
        # Get minimum dimensions
        min_w, min_h = (10, 10)  # Default
        for room_type, (w, h) in MIN_ROOM_SIZES.items():
            if room_type in room_lower:
                min_w, min_h = w, h
                break
        
        # Special cases
        if "master" in room_lower:
            min_w = int(min_w * 1.2)
            min_h = int(min_h * 1.2)
        elif "2-car garage" in room_lower:
            min_w, min_h = 20, 20
            
        # Check if room can fit
        if min_w > width or min_h > height:
            issues.append(f"{room} needs {min_w}x{min_h}ft but space is only {width}x{height}ft")
    
    # Check for too many rooms for the aspect ratio
    room_count = len(rooms)
    
    # Rough grid calculation
    if room_count > 6:
        # For many rooms, need roughly square-ish dimensions
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 1.5:
            issues.append(f"{room_count} rooms need more square dimensions (current {width}x{height}, aspect ratio {aspect_ratio:.1f})")
    
    # Minimum total dimensions for different room counts
    if room_count >= 6 and (width < 30 or height < 30):
        issues.append(f"{room_count} rooms need at least 30x30ft (current {width}x{height}ft)")
    elif room_count >= 8 and (width < 35 or height < 35):
        issues.append(f"{room_count} rooms need at least 35x35ft (current {width}x{height}ft)")
    
    return issues


def get_room_breakdown(rooms: List[str]) -> Dict[str, int]:
    """Get breakdown of room counts by type."""
    breakdown = {}
    for room in rooms:
        room_type = room.lower()
        # Normalize room types
        if "bedroom" in room_type:
            key = "bedrooms"
        elif "bathroom" in room_type:
            key = "bathrooms" 
        elif "garage" in room_type:
            key = "garages"
        else:
            key = room_type + "s"
            
        breakdown[key] = breakdown.get(key, 0) + 1
        
    return breakdown


def generate_suggestions(params: Dict, shortage: float, width: float, height: float) -> List[str]:
    """Generate suggestions to make layout feasible."""
    suggestions = []
    
    # Dimension suggestions
    new_width = width + math.ceil(shortage / (USABLE_SPACE_RATIO * height))
    new_height = height + math.ceil(shortage / (USABLE_SPACE_RATIO * width))
    suggestions.append(f"Increase to {new_width}x{height}ft or {width}x{new_height}ft")
    
    # Room reduction suggestions
    bedrooms = int(params.get("bedrooms", 2))
    if bedrooms > 2:
        suggestions.append(f"Reduce bedrooms from {bedrooms} to {bedrooms-1}")
    
    bathrooms = params.get("bathrooms", {})
    full_baths = int(bathrooms.get("full", 1))
    if full_baths > 1:
        suggestions.append(f"Reduce bathrooms from {full_baths} to {full_baths-1}")
        
    if params.get("garage", {}).get("attached", False):
        suggestions.append("Remove garage (saves 240+ sq ft)")
        
    if int(params.get("squareFeet", 1500)) >= 2000:
        suggestions.append("Reduce square footage target")
    
    return suggestions


if __name__ == "__main__":
    # Test with example parameters
    test_params = {
        "squareFeet": "2400",
        "bedrooms": "4",
        "bathrooms": {"full": "3", "half": "1"},
        "garage": {"attached": True, "cars": "2"},
        "office": True,
        "dimensions": {"width": 40, "depth": 40}
    }
    
    feasible, message, analysis = check_layout_feasibility(test_params)
    print(message)
    print(f"\nRoom breakdown: {analysis['room_breakdown']}")