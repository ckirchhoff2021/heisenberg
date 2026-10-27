"""typed-decisions -> kev 训练/评测格式。

typed-decisions 的 `prepared/{split}_questions.jsonl` 每条是 question 级：
  {id, case_id, workflow, state(JSON字符串), question{id,type,text,keys,candidates}, target[概率分布], label_semantics}

转成 kev 的 labelled request（带 soft target），再经 kev.data.materialize 得到内部 record。

题型映射（与 kev api.to_record 对齐）：
  jev choice  -> kev choice (criteria dict, key->描述)
  jev boolean -> kev noul   (criteria {"false":..,"true":..}；kev 内部 keys ["false","true"])
  jev score   -> kev score  (criteria = levels list；kev 内部 keys "0".."n-1")

label（materialize 需要，即使有 target）：
  choice -> option key（target argmax）
  noul   -> bool（target argmax == "true"）
  score  -> level index（target argmax）
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from kev.data import materialize


def load_typed_questions(path: str) -> list[dict]:
    """读 typed-decisions 的 questions.jsonl，返回 question 级 rows。"""
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def typed_to_request(rows_by_case: dict[str, list[dict]], case_id: str) -> dict:
    """一个 case 的 question 级 rows -> kev labelled request（带 soft target）。

    rows_by_case[case_id] 是该 case 的所有 question 级 rows（同一 state，5 个问题）。
    """
    qrows = rows_by_case[case_id]
    state = json.loads(qrows[0]["state"])  # state 是 JSON 字符串
    questions = {}
    for r in qrows:
        q = r["question"]
        kind = q["type"]
        target = r["target"]  # 概率分布，按 q["keys"] 顺序
        if kind == "choice":
            criteria = {k: c for k, c in zip(q["keys"], q["candidates"])}
            label = q["keys"][max(range(len(target)), key=lambda i: target[i])]
            tgt = {k: float(target[i]) for i, k in enumerate(q["keys"])}
        elif kind == "boolean":
            # jev keys ["true","false"], candidates=[true描述, false描述]
            criteria = {"false": q["candidates"][1], "true": q["candidates"][0]}
            p_true = float(target[0])  # target[0] 对应 "true"
            label = p_true >= 0.5
            tgt = {"false": float(target[1]), "true": p_true}
        else:  # score
            criteria = list(q["candidates"])
            label = int(max(range(len(target)), key=lambda i: target[i]))
            tgt = {str(i): float(target[i]) for i in range(len(target))}
        questions[q["id"]] = {
            "type": "noul" if kind == "boolean" else kind,
            "instructions": q["text"], "criteria": criteria,
            "label": label, "target": tgt, "src": r["workflow"],
        }
    return {"state": state, "questions": questions}


def group_by_case(rows: list[dict]) -> dict[str, list[dict]]:
    """按 case_id 分组（同一 case 的 5 个问题共享 state）。"""
    out = defaultdict(list)
    for r in rows:
        out[r["case_id"]].append(r)
    return out


def load_split(split: str, prepared_dir: str) -> list[dict]:
    """加载一个 split 的所有 case 的 labelled request（带 soft target）。"""
    path = Path(prepared_dir) / f"{split}_questions.jsonl"
    rows = load_typed_questions(str(path))
    by_case = group_by_case(rows)
    return [typed_to_request(by_case, cid) for cid in sorted(by_case)]


def load_split_materialized(split: str, prepared_dir: str) -> list[dict]:
    """加载并 materialize 一个 split，返回 kev 内部 record 列表（可直接 encode）。"""
    reqs = load_split(split, prepared_dir)
    return [materialize(r) for r in reqs]
