import zipfile

import config as cfg


def main():
    out = cfg.ROOT / "results_for_paper.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in cfg.RESULTS.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(cfg.ROOT))
        for name in ["analysis_dataset.csv", "coding_A.csv", "coding_B.csv"]:
            p = cfg.PROCESSED / name
            if p.exists():
                z.write(p, p.relative_to(cfg.ROOT))
        p = cfg.RAW / "spaces.json"
        if p.exists():
            z.write(p, p.relative_to(cfg.ROOT))
    print(f"created {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
