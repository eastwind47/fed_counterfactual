"""
generate_data.py
----------------
Root-level entry point for controlled-LTI data generation.

Usage:
    python generate_data.py

Configuration:
    Edit OUTPUT_PATH below. All system parameters come from configs/data_linear.yaml
    (read by data/synthetic_lti.py). The generator writes train/ and test/ splits plus
    dataset_params.json under OUTPUT_PATH. Run from the project root so that the
    `data.*` and `core.*` package imports resolve.
"""

from data.synthetic_lti import generate_and_save

# ── configure here ──────────────────────────────────────────────────────
OUTPUT_PATH = "data/datasets/linear_N6/"    # dataset root (gitignored)
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_and_save(OUTPUT_PATH)
