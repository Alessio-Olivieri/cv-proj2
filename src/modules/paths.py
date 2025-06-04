from pathlib import Path

print(__file__)
ROOT = Path(__file__).parent.parent.parent
DATA = Path(ROOT / "data")
