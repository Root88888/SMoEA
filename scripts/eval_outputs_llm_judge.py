#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/eval_outputs_llm_judge.py

【生成品質評測（LLM-as-a-judge）】對 main.py --mode batch 的逐筆輸出
（results/main_batch_outputs.jsonl）以 OpenAI 模型閱卷：每筆將
（題目, 標準答案, 模型輸出）交判官評分，回傳
  score       0–5 整數（5=完全正確）
  is_correct  布林（score≥4 視為正確）
  reason / major_errors  簡短評語與主要錯誤
標準答案自動自 dataset/test_data/task{N}_test.json 以 instance_id 對齊。

結果按三個維度彙總：整體、每任務（per_task）、路徑（per_path）。
routed 代表路由至單一 adapter；拒絕分支依 batch metadata 分成
rejected_base、rejected_artifact、rejected_arrow 或
rejected_taskwise_k16_arrow。舊版沒有 rejection_method 的輸出仍標成
rejected_merging；output 為 null 時列入 skipped_no_output，不送評。

【少量測試】（先跑 batch 產出結果檔，再評分）
  python main.py --mode batch --tasks 3,7 --limit 5
  export OPENAI_API_KEY=sk-...
  python scripts/eval_outputs_llm_judge.py --limit 5        # 每任務前 5 筆
  python scripts/eval_outputs_llm_judge.py --dry_run        # 不花錢：只驗資料對齊
【全量】直接不帶 --limit / --tasks；--resume 斷點續評。
輸出：results/llm_judge_results.json（含時間戳副本）。
"""

import argparse
import glob
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

JUDGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 5,
                  "description": "0 to 5. 5 means fully correct."},
        "is_correct": {"type": "boolean"},
        "reason": {"type": "string"},
        "major_errors": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "is_correct", "reason", "major_errors"],
    "additionalProperties": False,
}


def build_prompt(problem, reference, prediction):
    return f"""
You are a strict but fair evaluator for academic and STEM instruction-following tasks.

Judge whether the MODEL_PREDICTION is sufficiently correct compared with the REFERENCE_ANSWER for the given USER_QUERY.

Use a generous semantic grading standard:
- Accept different wording, order, formatting, or explanation style if the core answer is equivalent.
- Accept concise answers if they contain the essential final answer or key facts.
- Accept answers that are not identical to the reference but are scientifically/academically reasonable and answer the query.
- Do not require the model to copy the reference exactly.
- For open-ended explanation/advice/description tasks, mark correct if the prediction covers the main idea and does not contain major false claims.
- For classification, multiple-choice, diagnosis/category, or label tasks, the key label/category should match the reference.
- For math, physics, chemistry, finance, or other numeric tasks, the final numeric/symbolic answer should be equivalent after reasonable rounding, unit conversion, or algebraic reformulation.
- Small numerical rounding differences are acceptable.
- Large numerical differences, wrong powers of ten, wrong units that change the meaning, or wrong final options should be counted incorrect.
- If the prediction gives correct reasoning but the final answer is wrong, do not mark it fully correct.
- If the final answer is correct but the reasoning is shorter than the reference, it can still be correct.
- Penalize hallucinated facts, irrelevant content, contradictions, unsafe medical advice, or failure to answer the query.
- If the task asks for an exact string, sequence, molecule, formula, option, or named entity, the prediction must preserve the essential exact content.

Scoring:
5 = correct or essentially equivalent; minor wording/format/rounding differences only
4 = mostly correct; small omission or minor non-critical error
3 = partially correct; captures some key idea but misses important details
2 = weak; related but substantially incomplete or partly wrong
1 = mostly wrong but slightly related
0 = wrong, irrelevant, empty, or hallucinated

Set is_correct=true only for scores 4 or 5.

USER_QUERY:
{problem}

REFERENCE_ANSWER:
{reference}

MODEL_PREDICTION:
{prediction}
""".strip()


def clip(x, n=6000):
    x = "" if x is None else str(x).strip()
    return x if len(x) <= n else x[:n] + "\n...[TRUNCATED]..."


def extract_json_object(text):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    t2 = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    t2 = re.sub(r"```$", "", t2).strip()
    try:
        return json.loads(t2)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"Cannot parse JSON from: {text[:300]}")


def call_judge_api(client, model, prompt, max_tokens):
    """三層相容：Responses 結構化 → chat.completions → 裸 JSON。"""
    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "system",
                    "content": "You are an expert academic answer judge. "
                               "Output strict JSON only."},
                   {"role": "user", "content": prompt}],
            text={"format": {"type": "json_schema", "name": "llm_judge_result",
                             "schema": JUDGE_JSON_SCHEMA, "strict": True}},
            reasoning={"effort": "low"},
            max_output_tokens=max_tokens)
        return resp.output_text
    except TypeError:
        pass
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system",
                       "content": "You are an expert academic answer judge. "
                                  "Output strict JSON only."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "llm_judge_result",
                                             "schema": JUDGE_JSON_SCHEMA,
                                             "strict": True}},
            max_completion_tokens=max_tokens)
        return resp.choices[0].message.content
    except TypeError:
        resp = client.responses.create(
            model=model,
            input=[{"role": "system",
                    "content": "Output JSON only."},
                   {"role": "user",
                    "content": prompt + "\n\nReturn JSON only."}],
            max_output_tokens=max_tokens)
        return resp.output_text


NON_RETRYABLE = ("invalid_api_key", "Incorrect API key", "model_not_found",
                 "does not exist", "BadRequestError", "invalid_request_error")


def call_judge(client, model, item, max_retries, max_tokens):
    prompt = build_prompt(clip(item["input"]), clip(item["target"]),
                          clip(item["prediction"]))
    last_err = None
    for attempt in range(max_retries):
        try:
            parsed = extract_json_object(
                call_judge_api(client, model, prompt, max_tokens))
            parsed["score"] = max(0, min(5, int(parsed["score"])))
            parsed["is_correct"] = bool(parsed["is_correct"])
            return {**item, "judge": parsed, "error": None}
        except Exception as e:
            last_err = repr(e)
            if any(p in last_err for p in NON_RETRYABLE):
                break
            time.sleep(min(2 ** attempt, 10))
    return {**item, "judge": None, "error": last_err}


def load_references(dataset_dirs, tasks_needed):
    """依來源任務與 instance_id 載入所需標準答案。"""
    ref = {}
    for dataset_dir in dataset_dirs:
        if not dataset_dir:
            continue
        for t in sorted(tasks_needed):
            hits = sorted(glob.glob(os.path.join(dataset_dir,
                                                 f"{t}_test.json*")))
            for hit in hits:
                with open(hit, encoding="utf-8") as f:
                    obj = json.load(f)
                for x in (obj["instances"] if isinstance(obj, dict) else obj):
                    ref[(t, x["instance_id"])] = {
                        "input": x.get("full_prompt") or x.get("input", ""),
                        "target": x.get("output", "")}
    return ref


def build_items(batch_path, dataset_dir, tasks, limit, ood_dataset_dir=None):
    with open(batch_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if tasks:
        keep = {f"task{t.strip()}" for t in tasks.split(",")}
        rows = [r for r in rows if r["source_task"] in keep]
    if limit:
        per, out = defaultdict(int), []
        for r in rows:
            if per[r["source_task"]] < limit:
                out.append(r)
                per[r["source_task"]] += 1
        rows = out
    ref = load_references(
        [dataset_dir, ood_dataset_dir],
        {r["source_task"] for r in rows})

    items, skipped = [], []
    for r in rows:
        if r.get("routed_to"):
            path = "routed"
        elif r.get("rejection_method"):
            path = f"rejected_{r['rejection_method']}"
        else:
            path = "rejected_merging"
        base = {"instance_id": r["instance_id"],
                "source_task": r["source_task"],
                "routed_to": r.get("routed_to"),
                "path": path,
                "rejection_method": r.get("rejection_method"),
                "rejection_condition_id": r.get("rejection_condition_id"),
                "rejection_run_id": r.get("rejection_run_id")}
        g = ref.get((r["source_task"], r["instance_id"]))
        if r.get("output") is None:
            skipped.append({**base, "skip_reason": "no_output"})
        elif g is None:
            skipped.append({**base, "skip_reason": "no_reference"})
        else:
            items.append({**base, "input": g["input"], "target": g["target"],
                          "prediction": r["output"]})
    return items, skipped


def grp_summary(results):
    ok = [r for r in results if r.get("judge")]
    if not ok:
        return {"num_judged": 0, "accuracy": None, "avg_score": None}
    scores = [r["judge"]["score"] for r in ok]
    hist = {}
    for s in scores:
        hist[str(s)] = hist.get(str(s), 0) + 1
    return {"num_judged": len(ok),
            "accuracy": round(sum(r["judge"]["is_correct"] for r in ok)
                              / len(ok), 4),
            "avg_score": round(sum(scores) / len(scores), 3),
            "score_hist": dict(sorted(hist.items()))}


def summarize(results, skipped):
    by_task, by_path = defaultdict(list), defaultdict(list)
    for r in results:
        by_task[r["source_task"]].append(r)
        by_path[r["path"]].append(r)
    return {
        "overall": {**grp_summary(results),
                    "num_failed": sum(1 for r in results if r.get("error")),
                    "num_skipped_no_output":
                        sum(1 for s in skipped
                            if s["skip_reason"] == "no_output"),
                    "num_skipped_no_reference":
                        sum(1 for s in skipped
                            if s["skip_reason"] == "no_reference")},
        "per_path": {k: grp_summary(v) for k, v in sorted(by_path.items())},
        "per_task": {k: grp_summary(v) for k, v in sorted(by_task.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="results/main_batch_outputs.jsonl")
    ap.add_argument("--dataset_dir", default="dataset/test_data")
    ap.add_argument("--ood_dataset_dir", default="dataset/ood_test_data",
                    help="OOD 標準答案目錄；不存在時不影響一般 task 評測")
    ap.add_argument("--tasks", default=None,
                    help="僅評這些來源任務，如 3,7（預設全部）")
    ap.add_argument("--limit", type=int, default=None,
                    help="每任務最多評幾筆（少量測試用）")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--api_key", default=None,
                    help="預設讀環境變數 OPENAI_API_KEY")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--max_output_tokens", type=int, default=512)
    ap.add_argument("--out", default=None,
                    help="預設 results/llm_judge_{batch時間戳}.json——"
                         "檔名繼承所評 batch 檔的時間身分，與其時間戳副本"
                         "一眼配對；同一份 batch 重評落同一檔名")
    ap.add_argument("--resume", action="store_true",
                    help="跳過輸出檔中已成功評分的樣本")
    ap.add_argument("--dry_run", action="store_true",
                    help="不呼叫 API：只驗資料對齊並印統計骨架")
    ap.add_argument("--save_every", type=int, default=20)
    args = ap.parse_args()

    if args.out is None:
        m = re.search(r"(\d{8}_\d{6})", os.path.basename(args.batch))
        ts = m.group(1) if m else time.strftime(
            "%Y%m%d_%H%M%S", time.localtime(os.path.getmtime(args.batch)))
        args.out = os.path.join(os.path.dirname(args.batch) or ".",
                                f"llm_judge_{ts}.json")
    print(f"[judge] 評分對象 {args.batch} → 輸出 {args.out}")

    items, skipped = build_items(args.batch, args.dataset_dir,
                                 args.tasks, args.limit,
                                 args.ood_dataset_dir)
    n_task = len({x["source_task"] for x in items})
    print(f"[judge] 待評 {len(items)} 筆（{n_task} 任務）；"
          f"略過 {len(skipped)} 筆"
          f"（無輸出 {sum(1 for s in skipped if s['skip_reason']=='no_output')}"
          f"、無標準答案 "
          f"{sum(1 for s in skipped if s['skip_reason']=='no_reference')}）")

    if args.dry_run:
        for x in items[:3]:
            print(f"  例：{x['instance_id']} [{x['path']}] "
                  f"pred={x['prediction'][:60]!r} target={x['target'][:60]!r}")
        print("[dry_run] 資料對齊正常，未呼叫 API。")
        return

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    assert api_key, "請先 export OPENAI_API_KEY=...（或 --api_key）"
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    done = {}
    if args.resume and Path(args.out).exists():
        old = json.load(open(args.out, encoding="utf-8"))
        for r in old.get("results", []):
            if r.get("judge") is not None:
                done[r["instance_id"]] = r
        print(f"[resume] 已完成 {len(done)} 筆，續評其餘")
    todo = [x for x in items if x["instance_id"] not in done]
    results = list(done.values())

    def flush():
        h = hashlib.md5(open(args.batch, "rb").read()).hexdigest()[:12]
        payload = {"batch": args.batch, "batch_md5": h,
                   "model": args.model,
                   "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "summary": summarize(results, skipped),
                   "skipped": skipped,
                   "results": sorted(results,
                                     key=lambda r: r["instance_id"])}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)

    pending = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(call_judge, client, args.model, it,
                          args.max_retries, args.max_output_tokens)
                for it in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            pending += 1
            if pending >= args.save_every:
                flush()
                pending = 0
            if i % 20 == 0 or i == len(futs):
                print(f"  進度 {i}/{len(futs)}")
    flush()
    print(json.dumps(summarize(results, skipped)["overall"],
                     ensure_ascii=False, indent=1))
    print(f"[done] → {args.out}")


if __name__ == "__main__":
    main()
