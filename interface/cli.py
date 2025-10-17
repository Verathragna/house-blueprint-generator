import argparse
import base64
import json
import logging
import os
import sys
import time
import requests
from requests.exceptions import RequestException


def main():
    parser = argparse.ArgumentParser(description="Generate a house blueprint via the API")
    parser.add_argument(
        "--api",
        default="http://localhost:8000",
        help="Base URL of the Blueprint generation API",
    )
    parser.add_argument(
        "--params",
        default="sample_params.json",
        help="Path to JSON file with generation parameters",
    )
    parser.add_argument("--outdir", default="generated_cli", help="Directory to save outputs")
    parser.add_argument("--api-key", default="testkey", help="API key for authentication")
    parser.add_argument(
        "--strategy",
        default="greedy",
        choices=["greedy", "sample", "beam", "guided"],
        help="Decoding strategy",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--beam_size", type=int, default=5, help="Beam size for beam search"
    )
    parser.add_argument(
        "--min_separation",
        type=float,
        default=1.0,
        help="Minimum room separation; 0 disables",
    )
    parser.add_argument(
        "--guided_topk",
        type=int,
        default=8,
        help="Top-K expansion per step when using guided decoding",
    )
    parser.add_argument(
        "--guided_beam",
        type=int,
        default=8,
        help="Beam width when using guided decoding",
    )
    parser.add_argument(
        "--refine_iters",
        type=int,
        default=0,
        help="Iterations for post-decoding refinement (0 disables)",
    )
    parser.add_argument(
        "--refine_temp",
        type=float,
        default=5.0,
        help="Initial temperature for refinement search",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log = logging.getLogger(__name__)

    try:
        with open(args.params, "r", encoding="utf-8") as f:
            params = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.error("Failed to read parameters file %s: %s", args.params, e)
        sys.exit(1)

    payload = {
        "params": params,
        "strategy": args.strategy,
        "temperature": args.temperature,
        "beam_size": args.beam_size,
        "min_separation": args.min_separation,
    }
    if args.strategy == "guided":
        payload["guided_topk"] = args.guided_topk
        payload["guided_beam"] = args.guided_beam
    if args.refine_iters > 0:
        payload["refine_iterations"] = args.refine_iters
        payload["refine_temperature"] = args.refine_temp
    headers = {"X-API-Key": args.api_key}
    url = f"{args.api.rstrip('/')}/generate"
    try:
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    except RequestException as e:
        print(f"Request to {url} failed: {e}")
        return

    job_id = data["job_id"]
    status_url = f"{args.api.rstrip('/')}/status/{job_id}"
    while True:
        try:
            status_resp = requests.get(status_url, headers=headers)
            status_resp.raise_for_status()
            status_data = status_resp.json()
        except RequestException as e:
            print(f"Request to {status_url} failed: {e}")
            return
        if status_data["status"] == "completed":
            data = status_data["result"]
            break
        if status_data["status"] == "failed":
            print(f"Job failed: {status_data.get('error')}")
            return
        time.sleep(1)

    os.makedirs(args.outdir, exist_ok=True)
    svg_data = data["svg_data_url"].split(",", 1)[1]
    svg_bytes = base64.b64decode(svg_data)
    svg_path = os.path.join(args.outdir, data.get("svg_filename", "blueprint.svg"))
    try:
        with open(svg_path, "wb") as f:
            f.write(svg_bytes)
    except OSError as e:
        log.error("Failed to write SVG to %s: %s", svg_path, e)
        sys.exit(1)

    json_path = os.path.join(args.outdir, data.get("json_filename", "layout.json"))
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data["layout"], f, indent=2)
    except OSError as e:
        log.error("Failed to write layout JSON to %s: %s", json_path, e)
        sys.exit(1)

    gen_time = data.get("metadata", {}).get("processing_time")
    if gen_time is not None:
        print(f"Generation time: {gen_time:.2f}s")
    print(f"Saved SVG to {svg_path}")
    print(f"Saved layout JSON to {json_path}")


if __name__ == "__main__":
    main()
