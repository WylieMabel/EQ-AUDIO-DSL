import seisbench.data as sbd
import os
import sys

# Get the cache path to confirm it's correct
cache_root = os.environ.get("SEISBENCH_CACHE_ROOT", "~/.seisbench")
print(f"SeisBench will use cache root: {cache_root}")

try:
    # This line triggers the download to the new SEISBENCH_CACHE_ROOT
    print("Starting STEAD dataset check/download...")
    data = sbd.STEAD(force=False) # force=False is safer after the first attempt
    print(f"Dataset successfully loaded. Files should be in: {os.path.join(cache_root, 'datasets', 'stead')}")

except Exception as e:
    print(f"An error occurred during download: {e}")
    sys.exit(1)