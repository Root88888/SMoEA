#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/selftest_main_pipeline.py

【主程式管線自測】零 GPU、零模型——沿用端到端自測的合成資產，
把 main.py 批次模式完整走一遍：decide → 送審打分（偽 scorer）→
finalize → 按任務分組「生成」（stub 引擎）→ jsonl 落地。驗證：
  - 路由樣本使用 task adapter、拒絕樣本使用 selected rejection method；
  - 每行診斷鍵齊全（zone/margin/pval/top_units/top_tasks/…）；
  - stub 引擎收到的 adapter 切換與 routed_to 一致。

【執行】repo 根目錄：python scripts/selftest_main_pipeline.py
（真實生成路徑（載 base model＋adapter）不在本測範圍——那需要
GPU 與權重，由國網實測覆蓋。）
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from selftest_end_to_end import make_synth, WORDS  # noqa: E402
from system.merged_model import MergedModelError  # noqa: E402


class StubEngine:
    """記錄呼叫、回傳固定文本的假生成引擎。

    也模擬 registry 選擇（ADR-0001）：select_rejection 之後，ensure_rejection
    必須回報被選中的那一個，批次輸出才追溯得到來源。
    """

    def __init__(self, cfg):
        self.cfg = cfg["system"]
        self.switches = []
        self.selected = None

    def load_base(self):
        pass

    def ensure_adapter(self, task_key):
        self.switches.append(task_key)

    def available_rejections(self):
        return ()

    def current_rejection(self):
        return {"id": self.selected or "(config)", **self.ensure_identity()}

    def select_rejection(self, entry_id):
        if entry_id != "ties_only":
            raise MergedModelError(f"registry 沒有 id={entry_id!r} 的項目")
        self.selected = entry_id
        self.switches.append(f"select:{entry_id}")
        return self.current_rejection()

    def ensure_identity(self):
        if self.selected == "ties_only":
            return {"method": "artifact", "condition_id": "ties_only",
                    "run_id": "stub-run", "format": "dense_delta_v1"}
        return {"method": "base", "condition_id": "base", "run_id": None,
                "format": "base_model"}

    def ensure_rejection(self):
        self.switches.append("rejection")
        return self.ensure_identity()

    def generate(self, prompts):
        return [f"[stub:{self.switches[-1]}] {p[:24]}" for p in prompts]

    def unload(self):
        pass


def main():
    root = tempfile.mkdtemp(prefix="smoea_maintest_")
    cfgp, ad = make_synth(root)
    py = sys.executable
    r = subprocess.run([py, "scripts/build_router_assets.py",
                        "--config", cfgp], cwd=REPO,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    meta = json.load(open(os.path.join(ad, "router_assets_meta.json")))
    descs = {str(u): {"description": "synthetic unit "
             + " ".join(WORDS[meta["id_tasks"][g[0]]].split()[:3])}
             for u, g in enumerate(meta["units"])}
    with open(os.path.join(ad, "unit_descriptions.json"), "w",
              encoding="utf-8") as f:
        json.dump({"descriptions": descs}, f)

    import main as M
    from router.config import load_config
    from router.core import Router
    from router.verifier import make_fake_scorer

    def fake_swap_verify(cfg, rt, engine, texts, dec):
        rt.escalate(dec, texts, make_fake_scorer(),
                    rt.load_descriptions())

    M.swap_verify = fake_swap_verify
    M.InferenceEngine = StubEngine

    cfg = load_config(cfgp)
    rt = Router.load(cfg)
    M.run_batch(cfg, rt, tasks_arg=None, limit=None)

    out = os.path.join(root, "results", "main_batch_outputs.jsonl")
    lines = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert len(lines) == 6 * 40 + 2 * 40, f"行數 {len(lines)}"
    n_route = sum(1 for l in lines if l["routed_to"])
    n_rej = sum(1 for l in lines if l["routed_to"] is None)
    for l in lines:
        d = l["diagnosis"]
        for k in ("zone", "margin", "pval", "lex_agree",
                  "top_units", "top_tasks", "top_sims"):
            assert k in d, f"診斷缺鍵 {k}"
        if l["routed_to"] is None:
            assert l["output"].startswith("[stub:rejection]")
            assert l["model_source"] == "rejection"
            assert l["rejection_condition_id"] == "base"
            assert l["rejection_run_id"] is None
            assert d["zone"] in (2, 3)
        else:
            assert l["model_source"] == "task_adapter"
            assert l["rejection_condition_id"] is None
            assert l["rejection_run_id"] is None
            assert l["output"].startswith(f"[stub:{l['routed_to']}]"), \
                f"生成任務與路由不符：{l['routed_to']} vs {l['output'][:30]}"
    print(f"[MAIN-PASS] 批次 {len(lines)} 筆：路由 {n_route}／拒絕 {n_rej}"
          f"（rejection output 完整）；診斷鍵齊全；model source 與路由一致")

    # ---- registry 選擇（ADR-0001）：整批指定同一個 artifact ----
    M.run_batch(cfg, rt, tasks_arg=None, limit=None, artifact="ties_only")
    lines = [json.loads(l) for l in open(out, encoding="utf-8")]
    rejected = [l for l in lines if l["routed_to"] is None]
    assert rejected, "合成資料沒有拒絕樣本，無法驗證 artifact 選擇"
    for l in rejected:
        assert l["rejection_condition_id"] == "ties_only", l["rejection_condition_id"]
        assert l["rejection_run_id"] == "stub-run", l["rejection_run_id"]
        assert l["rejection_method"] == "artifact", l["rejection_method"]
    print(f"[SELECT-PASS] --artifact ties：{len(rejected)} 筆拒絕樣本"
          f"全部記錄為 ties_only:stub-run")

    # 選不存在的 id 必須失敗，而不是靜默退回 base
    try:
        M.run_batch(cfg, rt, tasks_arg=None, limit=None, artifact="不存在")
    except MergedModelError:
        print("[SELECT-PASS] 未知 id 正確被拒絕，未靜默退回 base")
    else:
        raise AssertionError("選了不存在的 id 卻沒有報錯")

    shutil.rmtree(root)
    print("ALL PASS（合成環境已清理）")


if __name__ == "__main__":
    main()
