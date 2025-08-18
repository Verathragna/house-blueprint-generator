import svgwrite

def render_layout_svg(layout_data, svg_path, scale=10):
    rooms = layout_data.get("layout", {}).get("rooms", [])

    # Determine canvas bounds based on room coordinates
    max_x = max((r.get("position", {}).get("x", 0) + r.get("size", {}).get("width", 0)) for r in rooms) * scale if rooms else 0
    max_y = max((r.get("position", {}).get("y", 0) + r.get("size", {}).get("length", 0)) for r in rooms) * scale if rooms else 0
    dwg = svgwrite.Drawing(svg_path, profile='tiny', size=(max_x, max_y))

    font_main = "Arial"
    font_size_label = 12
    font_size_sub = 10

    for room in rooms:
        x = room.get("position", {}).get("x", 0) * scale
        y = room.get("position", {}).get("y", 0) * scale
        w = room.get("size", {}).get("width", 12) * scale
        h = room.get("size", {}).get("length", 12) * scale

        dwg.add(dwg.rect(
            insert=(x, y),
            size=(w, h),
            fill="#ffffcc",
            stroke="black",
            stroke_width=2
        ))

        label_x = x + 5
        label_y = y + 15

        name = room.get("type", "Room")
        dwg.add(dwg.text(
            name,
            insert=(label_x, label_y),
            font_size=font_size_label,
            font_family=font_main,
            font_weight="bold"
        ))

        dims = f"{room.get('size', {}).get('width', 0)} x {room.get('size', {}).get('length', 0)}"
        dwg.add(dwg.text(
            dims,
            insert=(label_x, label_y + 14),
            font_size=font_size_sub,
            font_family=font_main
        ))

        ch = room.get("ceilingHeight")
        if ch:
            dwg.add(dwg.text(
                f"{ch}' Clg. Ht.",
                insert=(label_x, label_y + 28),
                font_size=font_size_sub,
                font_family=font_main
            ))

    dwg.save()
