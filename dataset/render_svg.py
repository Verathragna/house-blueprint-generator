import svgwrite

def render_layout_svg(layout_data, svg_path, scale=10):
    dwg = svgwrite.Drawing(svg_path, profile='tiny')

    font_main = "Arial"
    font_size_label = 12
    font_size_sub = 10

    for room in layout_data.get("layout", {}).get("rooms", []):
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
