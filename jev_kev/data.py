"""typed-decisions 数据管线（自包含，无 kev 依赖）。

既处理 typed-decisions -> labelled request（choice/noul/score 映射），
也提供 render / materialize，把 labelled request 转成 encode() 所需的
内部 record，行为与 kev 中的同名函数一致，但去掉了 pydantic 校验
（调用方就是本包数据管线，形状自证）。

labelled request：
  {"state": JSON 内容, "questions": {id: {type, instructions, criteria, label, src, target?}}}
    choice：criteria 为 {选项 key: 描述}，label 为选项 key
    noul  ：criteria 为 {"false":..,"true":..}，label 为 bool
    score ：criteria 为有序层级列表，label 为层级序号

materialize() -> 内部 record：
  {"state": str, "questions": [{"instr", "options", "label": int, "src",
                                "qtype", "qid", "keys", "target"?}]}
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load_typed_questions(path: str) -> list[dict]:
    """读 typed-decisions 的 questions.jsonl，返回 question 级 rows。"""
    return [json.loads(l)
            for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def render(v, indent: int = 0) -> str:
    """把 str | object | array 展平成模型看到的文本；字段名作为标签保留。"""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(
            f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(
        f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list))
        else f"{pad}{k}: {render(x)}" for k, x in v.items())


def option_text(name: str, desc) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def question_keys(qtype: str, criteria) -> list[str]:
    """每题概率/标签使用的 key，按选项顺序：
    choice=选项名；noul=["false","true"]；score=层级序号字符串。"""
    if qtype == "choice":
        return list(criteria)
    if qtype == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(criteria))]


def materialize(req: dict) -> dict:
    """labelled request -> 内部 record。

    label 转 int：choice=选项 key 在 keys 中的下标；noul=int(bool)；score=层级序号。
    若 question 带软标签 target（按 key 名给出的概率 dict），转成与 keys 对齐、
    归一化后的向量；未命名选项概率为 0。
    """
    questions = []
    for qid, src_q in req["questions"].items():
        qtype = src_q["type"]
        criteria = src_q.get("criteria")
        keys = question_keys(qtype, criteria)
        if qtype == "noul":
            c = criteria or {}
            options = [option_text("no", c.get("false")),
                       option_text("yes", c.get("true"))]
        elif qtype == "choice":
            options = [option_text(k, v) for k, v in criteria.items()]
        else:
            options = [render(x) for x in criteria]
        y = src_q["label"]
        label = keys.index(y) if qtype == "choice" else int(y)
        q = {
            "instr": render(src_q.get("instructions")),
            "options": options, "label": label,
            "src": src_q.get("src", ""), "qtype": qtype,
            "qid": qid, "keys": keys,
        }
        if src_q.get("target") is not None:
            t = [float(src_q["target"].get(k, 0.0)) for k in keys]
            if sum(t) <= 0:
                raise ValueError(f"target for {qid} puts no mass on any option")
            q["target"] = [x / sum(t) for x in t]
        questions.append(q)
    return {"state": render(req["state"]), "questions": questions}


def typed_to_request(rows_by_case: dict[str, list[dict]], case_id: str) -> dict:
    """一个 case 的 question 级 rows -> labelled request（带 soft target）。

    rows_by_case[case_id] 为该 case 全部 question 级 rows（同一 state，5 个问题）。
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
            # jev keys ["true","false"], candidates=[true 描述, false 描述]
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
    """加载并 materialize 一个 split，返回内部 record 列表（可直接 encode）。"""
    return [materialize(r) for r in load_split(split, prepared_dir)]
