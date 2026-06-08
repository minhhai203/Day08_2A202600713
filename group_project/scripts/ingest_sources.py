from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from agents.discovery_tools import list_available_sources


def main() -> None:
    source_root = Path(__file__).resolve().parent.parent / "data" / "source_docs"
    print(f"Source root: {source_root}")
    for name in list_available_sources():
        print(f"- {name}")


if __name__ == "__main__":
    main()
