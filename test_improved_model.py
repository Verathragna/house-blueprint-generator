import json
from Generate.generate_blueprint import generate_blueprint

# Test with parameters that previously failed
test_params = {
    'houseStyle': 'Modern',
    'squareFeet': '2000', 
    'bedrooms': '3',
    'bathrooms': {'full': '2'},
    'garage': {'attached': True}
}

print('Testing improved model with parameters that previously failed...')
try:
    result = generate_blueprint(test_params)
    if result and 'layout' in result:
        rooms = result['layout']['rooms']
        print(f'SUCCESS: Generated layout with {len(rooms)} rooms')
        for room in rooms:
            pos = room['position']
            size = room['size']
            print(f'  - {room["type"]}: ({pos["x"]}, {pos["y"]}) {size["width"]}x{size["length"]}')
    else:
        print('FAILED: No valid layout generated')
except Exception as e:
    print(f'ERROR: {e}')