import argparse
import base64
import json
import os
import requests


def main():
    parser = argparse.ArgumentParser(description="Generate a house blueprint via the API")
    parser.add_argument(
        "--api",
        default="http://localhost:8000/generate",
        help="Blueprint generation API endpoint",
    )
    parser.add_argument(
        "--params",
        default="sample_params.json",
        help="Path to JSON file with generation parameters",
    )
    parser.add_argument(
        "--outdir", default="generated_cli", help="Directory to save outputs"
    )
    args = parser.parse_args()

    with open(args.params, "r", encoding="utf-8") as f:
        params = json.load(f)

    try:
        resp = requests.post(args.api, json=params)
        data = resp.json()
    except Exception as e:
        print(f"Failed to contact API: {e}")
        return

    if resp.status_code != 200:
        print(f"Error: {data.get('message', data)}")
        return

    os.makedirs(args.outdir, exist_ok=True)
    svg_data = data["svg_data_url"].split(",", 1)[1]
    svg_bytes = base64.b64decode(svg_data)
    svg_path = os.path.join(args.outdir, data.get("svg_filename", "blueprint.svg"))
    with open(svg_path, "wb") as f:
        f.write(svg_bytes)

    json_path = os.path.join(args.outdir, data.get("json_filename", "layout.json"))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data["layout"], f, indent=2)

    gen_time = data.get("generation_time")
    if gen_time is not None:
        print(f"Generation time: {gen_time:.2f}s")
    print(f"Saved SVG to {svg_path}")
    print(f"Saved layout JSON to {json_path}")


if __name__ == "__main__":
    main()
