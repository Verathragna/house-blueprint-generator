#!/usr/bin/env python3
"""
Custom house generation using the working dataset approach
"""
import os, sys, json
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from dataset.realistic_layouts import RealisticLayoutGenerator
from dataset.render_svg import render_layout_svg

def generate_custom_house(params_file, output_prefix):
    """Generate a house layout using the working realistic layout generator"""
    
    # Load parameters
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    print(f"Generating house with parameters: {params}")
    
    # Extract key parameters
    width = params.get('dimensions', {}).get('width', 40)
    height = params.get('dimensions', {}).get('depth', 40) 
    bedrooms = int(params.get('bedrooms', 3))
    full_baths = int(params.get('bathrooms', {}).get('full', 2))
    half_baths = int(params.get('bathrooms', {}).get('half', 0))
    has_garage = params.get('garage', {}).get('attached', False) if isinstance(params.get('garage'), dict) else False
    
    # Generate using realistic layout generator (which works)
    generator = RealisticLayoutGenerator(
        max_width=width,
        max_height=height,
        target_density=0.5,
        min_spacing=2.0
    )
    
    layout_data = generator.generate_layout(params)
    
    if not layout_data:
        raise Exception("Failed to generate realistic layout")
    
    # Save JSON
    json_file = f"{output_prefix}.json"
    with open(json_file, 'w') as f:
        json.dump(layout_data, f, indent=2)
    
    # Render SVG
    svg_file = f"{output_prefix}.svg"
    render_layout_svg(layout_data, svg_file, lot_dims=(width, height))
    
    print(f"✅ Successfully generated: {json_file} and {svg_file}")
    return json_file, svg_file

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python custom_generate.py <params.json> <output_prefix>")
        sys.exit(1)
    
    params_file = sys.argv[1]
    output_prefix = sys.argv[2]
    
    try:
        generate_custom_house(params_file, output_prefix)
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        sys.exit(1)