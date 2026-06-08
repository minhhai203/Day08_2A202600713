from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from group_project.agents.corpus import iter_source_files, load_documents


def main() -> None:
    files = iter_source_files()
    if not files:
        print("Khong tim thay file nguon nao.")
        return

    print(f"Tim thay {len(files)} file nguon:")
    for f in files:
        print(f"  {f}")

    print()
    docs = load_documents()
    print(f"Da tai {len(docs)} document:")
    for doc in docs:
        meta = doc["metadata"]
        print(f"  [{meta['type']}] {meta['title']}  ->  {meta['source']}")


if __name__ == "__main__":
    main()

