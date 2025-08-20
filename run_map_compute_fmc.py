# run_map_compute_fmc.py

import argparse
from map_compute_fmc import map_compute_fmc

parser = argparse.ArgumentParser(description="Compute S2 LFMC maps from coordinates for the past month, including monthly mean")
parser.add_argument("--x", type=float, required=True, help="X coordinate")
parser.add_argument("--y", type=float, required=True, help="Y coordinate")

args = parser.parse_args()

map_compute_fmc(args.x, args.y)
