"""Record currently cached Hugging Face revisions without inferring run provenance."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parents[2]
CACHE = Path.home() / ".cache/huggingface/hub"
OUT = BASE / "experiments/checkpoint_revision_audit.md"
MODELS = [
    ("Chronos-Bolt-Base", "amazon", "chronos-bolt-base"),
    ("Chronos-2", "amazon", "chronos-2"),
    ("TimesFM-2.0-500M", "google", "timesfm-2.0-500m-pytorch"),
    ("Moirai-2-R-small", "Salesforce", "moirai-2.0-R-small"),
    ("TTM", "ibm", "TTM"),
]


def main():
    lines = ["# 本机模型缓存 revision 核查（2026-11-04）", "",
             "此报告只读本机 Hugging Face 缓存。运行脚本大多按仓库 ID 或 "
             "`revision=main` 加载模型；缓存当前只有一个 snapshot "
             "不等于能证明历史各行结果使用了该 revision。", "",
             "| 模型 | 当前 `refs/main` commit | 缓存快照数 | refs 文件本地修改时间（北京时间） |",
             "|---|---|---:|---|"]
    for label, owner, repo in MODELS:
        root = CACHE / f"models--{owner}--{repo}"
        ref = root / "refs/main"
        revision = ref.read_text().strip() if ref.exists() else "无"
        snapshots = list((root / "snapshots").iterdir()) if (root / "snapshots").exists() else []
        timestamp = (datetime.fromtimestamp(ref.stat().st_mtime, ZoneInfo("Asia/Shanghai"))
                     .strftime("%Y-%m-%d %H:%M:%S")) if ref.exists() else "无"
        lines.append(f"| {label} | `{revision}` | {len(snapshots)} | {timestamp} |")
    lines += ["", "`run_zeroshot.py`、`run_zeroshot_batched.py` 和 "
              "`run_finetune_ttm.py` 未统一在代码或结果行中记录这些 commit。 "
              "后续新实验应在每条运行记录中保存仓库 ID、resolved commit、" 
              "加载路径和依赖版本；当前旧归档只能标注 checkpoint revision 未核实。", ""]
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
