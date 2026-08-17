# -*- coding: utf-8 -*-
"""
router/core.py

【Router 類別】主程式與評測腳本共用的唯一路由邏輯所在。生命週期：

  建置   Router.build(cfg)      dataset + 嵌入快取 → 記憶體物件
         r.save()               落地全部資產（可檢視；線上物件同源）
  載入   Router.load(cfg)       從落地資產重建（不需 train 資料）
  推論   d = r.decide(texts)    嵌入→相似度→margin→p 值→四區（無 LLM、即時）
         r.escalate(d, texts, scorer, descs)
                                只對送審樣本打 LLM 是非題分（可批次/可快取）
         pred = r.finalize(d)   → task 索引陣列（-2=拒絕），直接餵 metrics

演算法（與封板系統逐步一致）：
  分數     全域 unit margin（top1−top2 單位相似度）
  直判     margin > transfer_floor → 免統計檢查直接路由
  p 值     Mondrian（按預測 top-1 單位分組）共形 p 值
  紅區     p < p_lo → 拒絕（誤拒率 ≤ p_lo 的定理預算）
  送審     p_lo ≤ p < p_hi，或綠區但詞彙 top-1 ≠ 嵌入 top-1
  綠區     其餘 → 路由
  裁決     送審樣本對全域 top-3 單位各問一題是非題，max p_yes ≥ θ 才路由
  還原     單位 → 任務：單位內任務相似度 argmax

【scale up】任務/單位數全由資產推導；資產檔結構隨任務數自動增長。
"""

import json
import os
import pickle

import numpy as np

from . import conformal, data_io, fingerprint, lexical, units as units_mod


ASSET_NPZ = "router_assets.npz"
ASSET_META = "router_assets_meta.json"
ASSET_LEX = "lexical_index.pkl"
ASSET_BANK = "calibration_bank.json"
ASSET_CENTS = "centroids.npz"
ASSET_CENTS_SUM = "centroids_summary.json"
ASSET_EXAMPLES = "unit_examples.json"
DESCRIPTIONS = "unit_descriptions.json"


class Router:
    def __init__(self, cfg):
        self.cfg = cfg
        self._embedder = None
        # 建置後成員：id_tasks, t_index, units, unit_of, FP, C, owner,
        # cal_b1, cal_margin, lex, unit_examples, merged_pairs, k_by_task

    # ------------------------------------------------------------------
    # 建置
    # ------------------------------------------------------------------
    @classmethod
    def build(cls, cfg, id_tasks):
        r = cls(cfg)
        r.id_tasks = list(id_tasks)
        r.t_index = {t: i for i, t in enumerate(r.id_tasks)}
        fracs = cfg["split"]["fractions"]
        seed = cfg["split"]["seed"]
        fpc = cfg["fingerprint"]

        means, cents, fp_txt, cal_list, r.k_by_task = {}, {}, {}, [], {}
        for t in r.id_tasks:
            emb = data_io.load_embeddings(cfg, t)
            texts = data_io.load_task_texts(cfg, t)
            if len(texts) != emb.shape[0]:
                raise ValueError(f"task{t} 文本數 {len(texts)} != 嵌入數 "
                                 f"{emb.shape[0]}（field 與快取不一致？）")
            i_fp, i_cal, _ = conformal.split_indices(
                emb.shape[0], fracs, seed + t)
            means[t] = fingerprint.task_fingerprint(emb[i_fp])
            cents[t], r.k_by_task[t] = fingerprint.select_centroids_inpile(
                emb[i_fp], fpc["k_max"], fpc["min_silhouette"],
                fpc["sil_sample"], seed)
            fp_txt[t] = [texts[i] for i in i_fp]
            cal_list.append(emb[i_cal])
            print(f"[build] task{t}: n={emb.shape[0]} k={r.k_by_task[t]}")

        r.FP = np.stack([means[t] for t in r.id_tasks])
        r.units, r.unit_of, r.merged_pairs = units_mod.build_units(
            r.FP, cfg["units"]["sim_threshold"])
        r.C, r.owner = fingerprint.stack_centroids(cents, r.id_tasks)
        print(f"[build] {len(r.id_tasks)} tasks -> {len(r.units)} units, "
              f"{r.C.shape[0]} centroids")

        # 校準分數（按第一名單位分組的全域 unit margin）
        Qc = np.concatenate(cal_list)
        tS = fingerprint.task_sims(Qc, r.C, r.owner, len(r.id_tasks))
        uS = fingerprint.unit_sims(tS, r.units)
        m = conformal.unit_margins(uS)
        r.cal_b1, r.cal_margin = m["b1"], m["margin"]

        # 詞彙索引（只 fit 指紋堆——可交換前提配套，勿改全量）
        r.lex = lexical.LexUnit(fp_txt, r.id_tasks, r.units,
                                cfg["lexical"]["tfidf_max_features"])

        # 送審裁決用的單位示例（每單位第一個任務的指紋堆第一筆）
        exm = cfg["verifier"]["example_max_chars"]
        r.unit_examples = {}
        from .verifier import clip_head_tail
        for u, g in enumerate(r.units):
            t = r.id_tasks[g[0]]
            r.unit_examples[str(u)] = clip_head_tail(
                fp_txt[t][0], exm * 2 // 3, exm // 3)
        return r

    # ------------------------------------------------------------------
    # 落地與載入
    # ------------------------------------------------------------------
    def save(self):
        ad = self.cfg["paths"]["assets_dir"]
        os.makedirs(ad, exist_ok=True)
        np.savez_compressed(
            os.path.join(ad, ASSET_NPZ),
            FP=self.FP, C=self.C, owner=self.owner, unit_of=self.unit_of,
            cal_b1=self.cal_b1, cal_margin=self.cal_margin)
        meta = {"id_tasks": self.id_tasks,
                "units": [[int(i) for i in g] for g in self.units],
                "k_by_task": {str(t): int(k)
                              for t, k in self.k_by_task.items()},
                "merged_pairs": self.merged_pairs,
                "n_calibration": int(self.cal_margin.size),
                "thresholds": self.cfg["thresholds"],
                "embedding_model": self.cfg["embedding"]["model_name"],
                "field": self.cfg["data"]["field"],
                "routing_text": self.cfg["data"].get(
                    "routing_text", self.cfg["data"]["field"])}
        with open(os.path.join(ad, ASSET_META), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        with open(os.path.join(ad, ASSET_LEX), "wb") as f:
            pickle.dump(self.lex, f)
        with open(os.path.join(ad, ASSET_EXAMPLES), "w",
                  encoding="utf-8") as f:
            json.dump(self.unit_examples, f, ensure_ascii=False, indent=1)
        # 可檢視資產：校準分數庫 + 質心
        bank = conformal.build_calibration_bank(
            self.cal_b1, self.cal_margin, self.units, self.id_tasks)
        with open(os.path.join(ad, ASSET_BANK), "w", encoding="utf-8") as f:
            json.dump(bank, f, ensure_ascii=False)
        cents, summary = {}, {}
        for ti, t in enumerate(self.id_tasks):
            cols = np.where(self.owner == ti)[0]
            cents[f"t{t}"] = self.C[cols]
            summary[f"t{t}"] = {"k": int(cols.size),
                                "multi": bool(cols.size > 1),
                                "unit": int(self.unit_of[ti])}
        np.savez_compressed(os.path.join(ad, ASSET_CENTS), **cents)
        with open(os.path.join(ad, ASSET_CENTS_SUM), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print(f"[save] 資產已落地 {ad}/（{ASSET_NPZ}, {ASSET_META}, "
              f"{ASSET_BANK}, {ASSET_CENTS}, {ASSET_LEX}, {ASSET_EXAMPLES}）")

    @classmethod
    def load(cls, cfg):
        ad = cfg["paths"]["assets_dir"]
        r = cls(cfg)
        z = np.load(os.path.join(ad, ASSET_NPZ))
        r.FP, r.C, r.owner = z["FP"], z["C"], z["owner"]
        r.unit_of = z["unit_of"]
        r.cal_b1, r.cal_margin = z["cal_b1"], z["cal_margin"]
        with open(os.path.join(ad, ASSET_META), encoding="utf-8") as f:
            meta = json.load(f)
        asset_routing_text = meta.get("routing_text", meta.get("field"))
        configured_routing_text = cfg["data"].get(
            "routing_text", cfg["data"]["field"])
        if (asset_routing_text is not None and
                asset_routing_text != configured_routing_text):
            raise ValueError(
                "router asset 的 routing_text="
                f"{asset_routing_text}，但設定為 {configured_routing_text}；"
                "請使用建置資產時相同的 data.routing_text")
        r.id_tasks = meta["id_tasks"]
        r.t_index = {t: i for i, t in enumerate(r.id_tasks)}
        r.units = [list(g) for g in meta["units"]]
        r.k_by_task = {int(t): k for t, k in meta["k_by_task"].items()}
        r.merged_pairs = meta.get("merged_pairs", [])
        with open(os.path.join(ad, ASSET_LEX), "rb") as f:
            r.lex = pickle.load(f)
        with open(os.path.join(ad, ASSET_EXAMPLES), encoding="utf-8") as f:
            r.unit_examples = json.load(f)
        return r

    def load_descriptions(self):
        """單位說明書（送審裁決輸入）。"""
        p = os.path.join(self.cfg["paths"]["assets_dir"], DESCRIPTIONS)
        with open(p, encoding="utf-8") as f:
            return json.load(f)["descriptions"]

    # ------------------------------------------------------------------
    # 推論
    # ------------------------------------------------------------------
    def _embed(self, texts):
        if self._embedder is None:
            from .embedding import Embedder
            self._embedder = Embedder(self.cfg)
        return self._embedder.encode(texts)

    def decide(self, texts, emb=None):
        """快路徑分區（無 LLM）。emb 給定時不重算嵌入（評測用快取）。

        回傳 dict of arrays：
          zone / b1(top-1 單位) / margin / pval / top3_units /
          lex_top1 / lex_disagree / tS(單位內還原用)
        """
        Q = emb if emb is not None else self._embed(texts)
        tS = fingerprint.task_sims(Q, self.C, self.owner, len(self.id_tasks))
        uS = fingerprint.unit_sims(tS, self.units)
        m = conformal.unit_margins(uS)
        pval = conformal.margin_pvals(m["b1"], m["margin"],
                                      self.cal_b1, self.cal_margin,
                                      len(self.units))
        lex_top1 = self.lex.top1_unit(texts)
        disagree = lex_top1 != m["b1"]
        if not self.cfg["thresholds"].get("use_lexical", True):
            disagree = np.zeros_like(disagree, dtype=bool)   # ablation：關閉詞彙訊號
        z = conformal.zones(m["margin"], pval, disagree,
                            self.cfg["thresholds"])
        return {"zone": z, "b1": m["b1"], "margin": m["margin"],
                "pval": pval, "top3_units": m["top"],
                "lex_top1": lex_top1, "lex_disagree": disagree, "tS": tS}

    def escalate(self, dec, texts, scorer, descriptions):
        """對 zone==escalate 的樣本打分。寫入 dec["esc"]：
        {row: {"units": [u1,u2,u3], "p_yes": [p1,p2,p3]}}。"""
        from .verifier import score_candidate
        esc = dec.setdefault("esc", {})
        for r in np.where(dec["zone"] == conformal.ZONE_ESCALATE)[0]:
            r = int(r)
            if r in esc:
                continue
            us, ps = [], []
            for u in dec["top3_units"][r]:
                u = int(u)
                us.append(u)
                ps.append(score_candidate(
                    scorer, descriptions[str(u)]["description"],
                    self.unit_examples.get(str(u), ""), texts[r]))
            esc[r] = {"units": us, "p_yes": ps}
        return dec

    def finalize(self, dec, count_missing=False, gray="verdict"):
        """分區＋裁決 → task 預測陣列（task 索引；-2=拒絕）。

        送審樣本無分數時視為拒絕（fail-closed）；count_missing=True 時
        另回傳缺分筆數供評測腳本斷言。
        gray（灰區/送審樣本的處置，ablation 用）：
          "verdict" 依 LLM 裁決（預設）| "reject" 全拒 | "route" 全依 top-1 放行。
        """
        theta = self.cfg["thresholds"]["theta_verify"]
        z = dec["zone"]
        pu = dec["b1"].copy()
        pu[z == conformal.ZONE_RED] = -2
        if gray == "reject":
            pu[z == conformal.ZONE_ESCALATE] = -2
        n_missing = 0
        esc = dec.get("esc", {})
        for r in (() if gray != "verdict"
                  else np.where(z == conformal.ZONE_ESCALATE)[0]):
            r = int(r)
            s = esc.get(r)
            if s is None:
                n_missing += 1
                pu[r] = -2
                continue
            j = int(np.argmax(s["p_yes"]))
            pu[r] = s["units"][j] if s["p_yes"][j] >= theta else -2
        # 單位 → 任務還原（單位內任務相似度 argmax）
        pt = pu.copy()
        ii = np.where(pu >= 0)[0]
        for u in np.unique(pu[ii]):
            g = self.units[int(u)]
            rows = ii[pu[ii] == u]
            pt[rows] = np.array(g)[dec["tS"][rows][:, g].argmax(axis=1)]
        return (pt, n_missing) if count_missing else pt
