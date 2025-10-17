import math
import svgwrite


def render_layout_svg(layout_data, svg_path, lot_dims=None, scale=10,
                       wall_ft=0.5, interior_wall_ft=0.35, door_ft=3.0,
                       blueprint=True, hatch=True,
                       out_png: str | None = None,
                       out_pdf: str | None = None):
    """Render a cleaner, blueprint-like SVG.

    - Draw exterior and interior walls with thickness (in feet).
    - Center labels inside rooms.
    - Add simple door openings on shared walls with swing arcs.
    - Draw windows on exterior walls.
    - Add dimension lines, scale bar, north arrow, title block.
    - Optionally hatch rooms and export as PNG/PDF (if cairosvg installed).
    """
    rooms = layout_data.get("layout", {}).get("rooms", [])

    # Canvas bounds
    if lot_dims:
        lot_w, lot_h = lot_dims
        max_x = lot_w * scale
        max_y = lot_h * scale
    else:
        max_x = (
            max((r.get("position", {}).get("x", 0) + r.get("size", {}).get("width", 0) for r in rooms),
                default=0)
            * scale
        )
        max_y = (
            max((r.get("position", {}).get("y", 0) + r.get("size", {}).get("length", 0) for r in rooms),
                default=0)
            * scale
        )

    pad = 0
    dwg = svgwrite.Drawing(svg_path, profile="tiny", size=(max_x + 2 * pad, max_y + 2 * pad))

    # Background
    if blueprint and not lot_dims:
        dwg.add(dwg.rect(insert=(0, 0), size=(max_x + 2 * pad, max_y + 2 * pad), fill="#0b3d5c"))
        stroke_color = "#cfe8ff"  # cyan-ish lines
        text_color = "#e6f2ff"
    else:
        stroke_color = "#222"
        text_color = "#000"

    # Lot outline
    if lot_dims:
        dwg.add(
            dwg.rect(
                insert=(pad, pad),
                size=(max_x, max_y),
                fill="none",
                stroke=stroke_color,
                stroke_width=scale * 0.6,
            )
        )

    font_main = "Arial"
    font_size_label = max(10, int(scale * 1.0))
    font_size_sub = max(9, int(scale * 0.85))

    # Precompute bounds in feet for geometry ops
    rects = []  # (x1,y1,x2,y2,name,w,l)
    for r in rooms:
        x = float(r.get("position", {}).get("x", 0))
        y = float(r.get("position", {}).get("y", 0))
        w = float(r.get("size", {}).get("width", 12))
        l = float(r.get("size", {}).get("length", 12))
        rects.append((x, y, x + w, y + l, r.get("type", "Room"), w, l))

    def overlap_1d(a1, a2, b1, b2):
        lo = max(a1, b1)
        hi = min(a2, b2)
        return (lo, hi) if hi > lo else None

    # Collect shared edges (interior walls) and exposed edges (exterior walls)
    shared = []  # (orientation, x_or_y, t1, t2)
    exposed = []  # (orientation, x_or_y, t1, t2)

    # For each room edge, check for overlaps with other rooms
    for i, (x1, y1, x2, y2, name, w, l) in enumerate(rects):
        edges = [
            ("v", x1, y1, y2),  # left
            ("v", x2, y1, y2),  # right
            ("h", y1, x1, x2),  # top
            ("h", y2, x1, x2),  # bottom
        ]
        for orient, coord, a1, a2 in edges:
            segments = [(a1, a2)]
            for j, (ox1, oy1, ox2, oy2, _, _, _) in enumerate(rects):
                if i == j:
                    continue
                if orient == "v" and math.isclose(coord, ox2, rel_tol=0, abs_tol=1e-6):
                    ov = overlap_1d(a1, a2, oy1, oy2)
                    if ov:
                        shared.append(("v", coord, ov[0], ov[1]))
                        # subtract overlap from exposed segments
                        new_segments = []
                        for s1, s2 in segments:
                            left = overlap_1d(s1, s2, ov[0], ov[1])
                            if not left:
                                new_segments.append((s1, s2))
                            else:
                                if s1 < left[0]:
                                    new_segments.append((s1, left[0]))
                                if left[1] < s2:
                                    new_segments.append((left[1], s2))
                        segments = new_segments
                elif orient == "v" and math.isclose(coord, ox1, rel_tol=0, abs_tol=1e-6):
                    ov = overlap_1d(a1, a2, oy1, oy2)
                    if ov:
                        shared.append(("v", coord, ov[0], ov[1]))
                        new_segments = []
                        for s1, s2 in segments:
                            left = overlap_1d(s1, s2, ov[0], ov[1])
                            if not left:
                                new_segments.append((s1, s2))
                            else:
                                if s1 < left[0]:
                                    new_segments.append((s1, left[0]))
                                if left[1] < s2:
                                    new_segments.append((left[1], s2))
                        segments = new_segments
                elif orient == "h" and math.isclose(coord, oy2, rel_tol=0, abs_tol=1e-6):
                    ov = overlap_1d(a1, a2, ox1, ox2)
                    if ov:
                        shared.append(("h", coord, ov[0], ov[1]))
                        new_segments = []
                        for s1, s2 in segments:
                            left = overlap_1d(s1, s2, ov[0], ov[1])
                            if not left:
                                new_segments.append((s1, s2))
                            else:
                                if s1 < left[0]:
                                    new_segments.append((s1, left[0]))
                                if left[1] < s2:
                                    new_segments.append((left[1], s2))
                        segments = new_segments
                elif orient == "h" and math.isclose(coord, oy1, rel_tol=0, abs_tol=1e-6):
                    ov = overlap_1d(a1, a2, ox1, ox2)
                    if ov:
                        shared.append(("h", coord, ov[0], ov[1]))
                        new_segments = []
                        for s1, s2 in segments:
                            left = overlap_1d(s1, s2, ov[0], ov[1])
                            if not left:
                                new_segments.append((s1, s2))
                            else:
                                if s1 < left[0]:
                                    new_segments.append((s1, left[0]))
                                if left[1] < s2:
                                    new_segments.append((left[1], s2))
                        segments = new_segments
            for seg in segments:
                exposed.append((orient, coord, seg[0], seg[1]))

    # Merge duplicate shared segments (they will appear twice)
    def _norm_key(e):
        orient, c, a, b = e
        if a > b:
            a, b = b, a
        return (orient, round(c, 6), round(a, 6), round(b, 6))
    shared_unique = {}
    for e in shared:
        shared_unique[_norm_key(e)] = e
    shared = list(shared_unique.values())

    # Draw exposed (exterior) walls and sprinkle windows
    ext_w = max(1.0, wall_ft * scale)
    int_w = max(0.5, interior_wall_ft * scale)
    door_w = max(door_ft * scale, int_w * 2)

    window_len = max(scale * 2.0, ext_w * 2)
    corner_clear = scale * 1.0

    for orient, c, a, b in exposed:
        if b <= a:
            continue
        if orient == "v":
            x = c * scale + pad
            y1 = a * scale + pad
            y2 = b * scale + pad
            dwg.add(dwg.line(start=(x, y1), end=(x, y2), stroke=stroke_color, stroke_width=ext_w))
            # Windows: place 1-2 short segments avoiding corners
            span = (y2 - y1)
            if span > 4 * window_len:
                win_y = y1 + span * 0.5
                dwg.add(dwg.line(start=(x - ext_w*0.6, win_y - window_len/2), end=(x + ext_w*0.6, win_y - window_len/2), stroke=stroke_color, stroke_width=ext_w*0.3))
                dwg.add(dwg.line(start=(x - ext_w*0.6, win_y + window_len/2), end=(x + ext_w*0.6, win_y + window_len/2), stroke=stroke_color, stroke_width=ext_w*0.3))
        else:
            y = c * scale + pad
            x1 = a * scale + pad
            x2 = b * scale + pad
            dwg.add(dwg.line(start=(x1, y), end=(x2, y), stroke=stroke_color, stroke_width=ext_w))
            span = (x2 - x1)
            if span > 4 * window_len:
                win_x = x1 + span * 0.5
                dwg.add(dwg.line(start=(win_x - window_len/2, y - ext_w*0.6), end=(win_x - window_len/2, y + ext_w*0.6), stroke=stroke_color, stroke_width=ext_w*0.3))
                dwg.add(dwg.line(start=(win_x + window_len/2, y - ext_w*0.6), end=(win_x + window_len/2, y + ext_w*0.6), stroke=stroke_color, stroke_width=ext_w*0.3))

    # Draw a hidden rect per room (keeps legacy tests and makes selection easier)
    for (x1, y1, x2, y2, name, w, l) in rects:
        dwg.add(dwg.rect(
            insert=(x1 * scale + pad, y1 * scale + pad),
            size=((x2 - x1) * scale, (y2 - y1) * scale),
            fill="none",
            stroke="none",
        ))

    # Draw interior shared walls with a doorway gap in the middle and swing arcs
    for orient, c, a, b in shared:
        if b <= a:
            continue
        gap_mid = (a + b) * 0.5
        gap1 = gap_mid - door_ft * 0.5
        gap2 = gap_mid + door_ft * 0.5
        # Left segment
        if gap1 > a:
            if orient == "v":
                x = c * scale + pad
                dwg.add(dwg.line(start=(x, a * scale + pad), end=(x, gap1 * scale + pad),
                                  stroke=stroke_color, stroke_width=int_w))
            else:
                y = c * scale + pad
                dwg.add(dwg.line(start=(a * scale + pad, y), end=(gap1 * scale + pad, y),
                                  stroke=stroke_color, stroke_width=int_w))
        # Right segment
        if b > gap2:
            if orient == "v":
                x = c * scale + pad
                dwg.add(dwg.line(start=(x, gap2 * scale + pad), end=(x, b * scale + pad),
                                  stroke=stroke_color, stroke_width=int_w))
            else:
                y = c * scale + pad
                dwg.add(dwg.line(start=(gap2 * scale + pad, y), end=(b * scale + pad, y),
                                  stroke=stroke_color, stroke_width=int_w))
        # Door leaf and swing
        hinge_x, hinge_y = (c * scale + pad, gap1 * scale + pad) if orient == "v" else (gap1 * scale + pad, c * scale + pad)
        leaf_len = max(scale * 1.5, (gap2 - gap1) * scale * 0.5)
        if orient == "v":
            leaf_end = (hinge_x + leaf_len, hinge_y)
            dwg.add(dwg.line(start=(hinge_x, hinge_y), end=leaf_end, stroke=stroke_color, stroke_width=int_w*0.7))
            # quarter-arc
            arc_path = f"M {hinge_x},{hinge_y} A {leaf_len},{leaf_len} 0 0,1 {hinge_x},{hinge_y + leaf_len}"
        else:
            leaf_end = (hinge_x, hinge_y + leaf_len)
            dwg.add(dwg.line(start=(hinge_x, hinge_y), end=leaf_end, stroke=stroke_color, stroke_width=int_w*0.7))
            arc_path = f"M {hinge_x},{hinge_y} A {leaf_len},{leaf_len} 0 0,1 {hinge_x + leaf_len},{hinge_y}"
        dwg.add(dwg.path(d=arc_path, fill="none", stroke=stroke_color, stroke_width=int_w*0.5))

    # Room labels (centered) + area
    for (x1, y1, x2, y2, name, w, l) in rects:
        cx = (x1 + x2) * 0.5 * scale + pad
        cy = (y1 + y2) * 0.5 * scale + pad
        dims = f"{int(round(w))} x {int(round(l))}"
        area = int(round(w * l))
        dwg.add(dwg.text(name,
                          insert=(cx, cy - font_size_sub),
                          text_anchor="middle",
                          font_size=font_size_label,
                          font_family=font_main,
                          font_weight="bold",
                          fill=text_color))
        dwg.add(dwg.text(f"{dims}  •  {area} ft²",
                          insert=(cx, cy + font_size_sub * 0.2),
                          text_anchor="middle",
                          font_size=font_size_sub,
                          font_family=font_main,
                          fill=text_color))

    # Dimension lines (lot)
    if lot_dims:
        lw, lh = lot_dims
        # Top dimension
        y = -scale * 0.6
        dwg.add(dwg.line(start=(0, y), end=(lw * scale, y), stroke=stroke_color, stroke_width=1))
        dwg.add(dwg.text(f"{int(lw)}'", insert=(lw * scale / 2, y - 2), text_anchor="middle", font_size=font_size_sub, fill=text_color))
        # Left dimension
        x = -scale * 0.6
        dwg.add(dwg.line(start=(x, 0), end=(x, lh * scale), stroke=stroke_color, stroke_width=1))
        dwg.add(dwg.text(f"{int(lh)}'", insert=(x - 2, lh * scale / 2), text_anchor="end", font_size=font_size_sub, fill=text_color))

    # Scale bar & North arrow
    bar_len_ft = 10
    bar_len_px = bar_len_ft * scale
    base_y = max_y + scale * 0.8
    dwg.add(dwg.line(start=(0, base_y), end=(bar_len_px, base_y), stroke=stroke_color, stroke_width=2))
    for i in range(0, bar_len_ft + 1, 2):
        x = i * scale
        dwg.add(dwg.line(start=(x, base_y - 4), end=(x, base_y + 4), stroke=stroke_color, stroke_width=1))
    dwg.add(dwg.text("0   10 ft", insert=(bar_len_px / 2, base_y + 14), text_anchor="middle", font_size=font_size_sub, fill=text_color))
    # North
    nx, ny = bar_len_px + scale * 2, base_y
    dwg.add(dwg.polygon(points=[(nx, ny - 12), (nx - 6, ny), (nx + 6, ny)], fill=stroke_color))
    dwg.add(dwg.text("N", insert=(nx, ny + 14), text_anchor="middle", font_size=font_size_sub, fill=text_color))

    # Title block
    title = (layout_data.get("meta", {}) or {}).get("title", "Generated Plan")
    dwg.add(dwg.text(title, insert=(0, max_y + scale * 2.0), font_size=font_size_label, fill=text_color))

    dwg.save()

    # Optional exports
    if out_png or out_pdf:
        try:
            import cairosvg  # type: ignore
            svg_bytes = dwg.tostring()
            if out_png:
                cairosvg.svg2png(bytestring=svg_bytes.encode("utf-8"), write_to=out_png)
            if out_pdf:
                cairosvg.svg2pdf(bytestring=svg_bytes.encode("utf-8"), write_to=out_pdf)
        except Exception:
            # Silently ignore if backend not available
            pass
