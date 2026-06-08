from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
THANH_STANDARDIZED = REPO_ROOT / "personal_project" / "2A202600838_NguyenDucThanh" / "data" / "standardized"
GROUP_SOURCE_DOCS = REPO_ROOT / "group_project" / "data" / "source_docs"


def ingest() -> None:
    if not THANH_STANDARDIZED.exists():
        print(f"[ingest] Source not found: {THANH_STANDARDIZED}")
        sys.exit(1)

    copied, skipped = 0, 0
    for src in sorted(THANH_STANDARDIZED.rglob("*.md")):
        subfolder = src.parent.name  # "legal" or "news"
        dest_dir = GROUP_SOURCE_DOCS / subfolder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name

        existing_stems = {p.stem for p in dest_dir.glob("*.md")}
        if src.stem in existing_stems:
            print(f"  [skip]  {subfolder}/{src.name}  (stem already exists)")
            skipped += 1
            continue

        shutil.copy2(src, dest)
        print(f"  [copy]  {subfolder}/{src.name}  -> {dest.relative_to(REPO_ROOT)}")
        copied += 1

    print(f"\nDone: {copied} copied, {skipped} skipped.")
    _verify()


def _verify() -> None:
    sys.path.insert(0, str(REPO_ROOT / "group_project"))
    from agents.corpus import load_documents

    docs = load_documents()
    print(f"\n[verify] corpus loaded {len(docs)} document(s):")
    for doc in docs:
        meta = doc.get("metadata", {})
        title = (meta.get("title") or "?").encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
        print(f"  - {title}  ({meta.get('type', '?')})")


if __name__ == "__main__":
    ingest()
