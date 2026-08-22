import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluation.latency import run_latency_benchmark

if __name__ == "__main__":
    run_latency_benchmark()
