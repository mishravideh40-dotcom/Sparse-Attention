"""Download the TinyShakespeare dataset used by scripts/quality_eval.py.

data/ is gitignored (it's a reproducible download, not project source), so
this script is how a grader regenerates it.
"""
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "tinyshakespeare.txt"

if __name__ == "__main__":
    OUT_PATH.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(URL, OUT_PATH)
    size = OUT_PATH.stat().st_size
    print(f"Downloaded {size} bytes to {OUT_PATH}")
