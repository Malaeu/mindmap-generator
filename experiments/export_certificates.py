#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from glob import glob


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    out_md = sys.argv[1] if len(sys.argv) > 1 else "reports/CERTIFICATES.md"
    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    files = sorted(glob("certs/*.json"))
    lines = []
    lines.append("# Certified Validation Summaries\n")
    for path in files:
        data = load_json(path)
        lines.append(f"\n## {os.path.basename(path)}\n")
        lines.append(f"Family: `{data.get('family')}`; zeros: {data.get('zeros')}; nodes: {len(data.get('nodes', []))}\n")
        lines.append("| sigma | Q_lower | Q_upper | Z | A | P |\n")
        lines.append("|------:|-------:|-------:|---:|---:|---:|\n")
        for n in data.get("nodes", []):
            s = n["sigma"]
            z = n["Z"]
            a = n["A"]
            p = n["P"]
            lines.append(
                f"| {s:.6g} | {n['Q_lower']:+.6e} | {n['Q_upper']:+.6e} | "
                f"[{z[0]:+.3e},{z[1]:+.3e}] | [{a[0]:+.3e},{a[1]:+.3e}] | [{p[0]:+.3e},{p[1]:+.3e}] |"
            )
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()

