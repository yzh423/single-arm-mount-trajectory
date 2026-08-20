from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doosan_teleop.offline_consequence import view_main

if __name__ == "__main__":
    raise SystemExit(view_main("easyik"))
