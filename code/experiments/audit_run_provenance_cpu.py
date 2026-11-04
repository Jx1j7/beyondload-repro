"""Match archived step=1 result rows to existing logs and backup CSVs.

This is read-only provenance reconstruction, not a claim that CSV `date` is an
execution timestamp. Each log match uses model, dataset, seed, and rounded MAE.
"""
from collections import Counter
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
EXP = BASE / "experiments"
OUT = EXP / "provenance_audit"
LOGS = ["cloud_run_all.log", "main_allzones.log", "backfill.log", "zone3_redo.log"]
BACKUPS = ["results_local_backup_1102.csv", "cloud_results.csv", "cloud_results_final.csv"]


def log_rows(path: Path, models: set[str]) -> set[tuple]:
    found = set()
    for line in path.read_text(errors="replace").splitlines():
        tokens = line.split()
        if len(tokens) < 12 or tokens[0] not in models:
            continue
        try:
            index = next(i for i, token in enumerate(tokens)
                         if token.startswith("gefcom2014_"))
            found.add((tokens[0], tokens[index], int(float(tokens[index + 2])),
                       round(float(tokens[1]), 4)))
        except (StopIteration, ValueError, IndexError):
            continue
    return found


def csv_rows(path: Path) -> set[tuple]:
    frame = pd.read_csv(path)
    return {(r.model, r.dataset, int(r.seed), round(float(r.MAE), 6))
            for r in frame.itertuples() if pd.notna(r.MAE)}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    frame = pd.read_csv(EXP / "results.csv")
    frame = frame[frame.note.fillna("").str.contains(r"(?:^|;)\s*step=1(?:;|$)", regex=True)]
    frame = frame.drop_duplicates(["model", "dataset", "seed"], keep="last").copy()
    logs = {name: log_rows(EXP / name, set(frame.model)) for name in LOGS}
    backups = {name: csv_rows(EXP / name) for name in BACKUPS}
    evidence = []
    for r in frame.itertuples():
        key4 = (r.model, r.dataset, int(r.seed), round(float(r.MAE), 4))
        key6 = (r.model, r.dataset, int(r.seed), round(float(r.MAE), 6))
        matched_logs = [name for name, rows in logs.items() if key4 in rows]
        matched_backups = [name for name, rows in backups.items() if key6 in rows]
        evidence.append(dict(model=r.model, dataset=r.dataset, seed=int(r.seed),
                             mae=float(r.MAE), csv_date=r.date,
                             log_matches=";".join(matched_logs),
                             backup_matches=";".join(matched_backups),
                             has_log_match=bool(matched_logs)))
    result = pd.DataFrame(evidence)
    result.to_csv(OUT / "step1_run_evidence.csv", index=False)
    count = Counter()
    for item in evidence:
        names = tuple(item["log_matches"].split(";")) if item["log_matches"] else ()
        count[names] += 1
    lines = ["# 主结果运行时间线：现有记录重建", "",
             "本报告不重新运行模型，不以 CSV `date` 字段推断实际执行时间：多个运行脚本将该字段写为固定字符串。",
             "", f"- 去重 `step=1` 主结果：{len(result)} 条。",
             f"- 有模型、站点、种子、四位小数 MAE 一致的日志记录：{int(result.has_log_match.sum())} 条。",
             f"- 无这类日志记录：{int((~result.has_log_match).sum())} 条；见逐行证据表。",
             "- `cloud_run_all.log` 本地文件修改时间为 2026-11-02 22:05（Asia/Shanghai），其匹配结果至迟在协议 11-03 冻结日期之前已存在。",
             "- `main_allzones.log` 末尾写有实际完成标记：2026-11-03 17:41:38 UTC，即北京时间 2026-11-04 01:41:38。该日志记录的行可证明属于这一批或此前已存在、被断点脚本跳过的结果；不能仅凭出现于该日志认定它首次运行于冻结之后。",
             "- `zone3_redo.log`、`backfill.log` 和最终备份提供后续修复痕迹；准确到每条运行的开始时刻需更多原始 shell/会话证据。",
             "", "## 精确日志匹配组合", "",
             "| 记录数 | 匹配日志 |", "|---:|---|"]
    for names, n in count.most_common():
        lines.append(f"| {n} | {', '.join(names) if names else '无'} |")
    unmatched = result.loc[~result.has_log_match]
    old_backup = unmatched.backup_matches.str.contains(
        "results_local_backup_1102.csv|cloud_results.csv", regex=True)
    final_only = unmatched.backup_matches.eq("cloud_results_final.csv")
    no_backup = unmatched.backup_matches.eq("")
    lines += ["", "## 十条无精确日志匹配记录的归类", "",
              f"- {int(old_backup.sum())} 条可在 `results_local_backup_1102.csv` 或 `cloud_results.csv` 中以六位小数 MAE 匹配。两份文件的本地修改时间分别为 2026-11-02 22:40 和 21:54（北京时间），是冻结前已有结果的工件证据，不提供逐条首跑时刻。",
              f"- {int(final_only.sum())} 条只匹配 2026-11-04 的 `cloud_results_final.csv`；首次运行时间未知。",
              f"- {int(no_backup.sum())} 条没有匹配的日志或备份；来源时间未知。",
              "", "| 模型 | 数据集 | 种子 | 已有备份 |", "|---|---|---:|---|"]
    for row in unmatched.itertuples():
        lines.append(f"| {row.model} | {row.dataset} | {row.seed} | {row.backup_matches or '无'} |")
    lines += ["", "## 论文表述边界", "",
              "现有证据显示主结果混合了冻结前已有记录和冻结后主批次/修复批次。无需重跑即可如实披露。当前不能把所有 397 条结果统称为预注册后运行；使用“版本化协议、回顾性纳入清单、探索性/确认性结果分开报告”更准确。",
              "", "逐行证据：`step1_run_evidence.csv`。本表给出已有工件的匹配，不伪造缺失的精确时间戳。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:15]))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
