#!/usr/bin/env python
"""
Emit the versioned JSON Schema for Params to schema/params.v1.json
"""
import os
from Generate.orchestrator import emit_params_schema

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "schema", "params.v1.json")

if __name__ == "__main__":
    emit_params_schema(OUT)
    print(f"Wrote {OUT}")
