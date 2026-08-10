#!/usr/bin/env bash
# =============================================================================
# scripts/setup_workspace.sh — 本機一鍵前置（Ubuntu ＋ NVIDIA GPU ＋ conda）
#
# 用途：clone 本 repo、把「需自行準備」的檔案放好之後，在 repo 根目錄執行
# 本腳本；結束時 main.py 即可直接使用（例如測試 system/rejection.py）。
# 可重複執行：已完成的步驟自動跳過。router 的離線準備全部由本腳本代辦。
#
#   bash scripts/setup_workspace.sh
#
# ─────────────────────────────────────────────────────────────────────────────
# 【需自行準備】（腳本會逐項檢查，缺哪項會明講）
#   1. conda（miniconda 即可）與 NVIDIA 驅動已安裝
#   2. 任務樣本 → dataset/train_data/、dataset/test_data/
#        task{N}_train.json / task{N}_test.json
#   3. 單位說明書 → assets/unit_descriptions.json
#   4. adapters → adapter/task{N}/（自 Google Drive 下載解壓後攤平放入；
#        每個任務一個目錄，內含 adapter 檔或 checkpoint-*/）
#
# 【腳本代辦】conda env 建置＋鎖定依賴、環境體檢、查詢嵌入計算
# （首次自動下載嵌入模型 ~1.3GB；GPU 數分鐘）、路由資產建置。
# base model 與裁決模型（各 ~16GB）會在首次執行 main.py 時自動下載。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== SMoEA workspace setup @ $(pwd) =="

fail() { echo "✗ $1"; echo "  → $2"; exit 1; }

# ---- 0/4 檢查「需自行準備」清單 ----
command -v conda >/dev/null || fail "找不到 conda" "安裝 miniconda 後重跑"
[ "$(ls dataset/train_data/task*.json* 2>/dev/null | wc -l)" -ge 1 ] \
  || fail "dataset/train_data/ 沒有任務樣本"
[ "$(ls dataset/test_data/task*.json* 2>/dev/null | wc -l)" -ge 1 ] \
  || fail "dataset/test_data/ 沒有測試樣本"
[ -f assets/unit_descriptions.json ] \
  || fail "缺 assets/unit_descriptions.json"
[ "$(ls -d adapter/task* 2>/dev/null | wc -l)" -ge 1 ] \
  || fail "adapter/ 沒有任務目錄" "自 Google Drive 下載解壓後放入"
echo "[0/4] 需自行準備的檔案齊全"
echo "      樣本 train $(ls dataset/train_data | wc -l) / test $(ls dataset/test_data | wc -l) 檔、adapter $(ls -d adapter/task* | wc -l) 個任務"

# ---- 1/4 conda env（不存在則建置；名稱 smoea）----
source "$(conda info --base)/etc/profile.d/conda.sh"
while [ -n "${CONDA_DEFAULT_ENV:-}" ]; do conda deactivate; done
if ! conda env list | grep -qE '^smoea\s'; then
    echo "[1/4] 建置 conda env smoea（首次含 torch 下載，約 10-20 分鐘）…"
    conda create -y -n smoea python=3.12
    conda activate smoea
    export PYTHONNOUSERSITE=1
    python -m pip install -r requirements-lock-twcc.txt
else
    echo "[1/4] conda env smoea 已存在，跳過建置"
    conda activate smoea
    export PYTHONNOUSERSITE=1
fi

# ---- 2/4 環境體檢 ----
echo "[2/4] 環境體檢"
python scripts/check_env.py || fail "體檢未過" "照上方 FAIL 提示處置後重跑本腳本"

# ---- 3/4 路由資產（缺則建置；含查詢嵌入計算，快取後不重算）----
mkdir -p results
if [ ! -f assets/router_assets.npz ]; then
    echo "[3/4] 建置路由資產（首次含嵌入計算；GPU 數分鐘）…"
    python scripts/build_router_assets.py
else
    echo "[3/4] 路由資產已存在，跳過（要重建：先刪 assets/router_assets.npz）"
fi

# ---- 4/4 完成 ----
echo
echo "== 全部就緒 =="
echo "之後每次開終端機："
echo "  conda activate smoea"
echo "測試（互動模式；輸入域外 query 會進 system/rejection.py；"
echo "首次執行會自動下載 base model 與裁決模型，各 ~16GB）："
echo "  python main.py --mode interactive"
echo "批次（拒絕樣本輸出落在 results/main_batch_outputs.jsonl）："
echo "  python main.py --mode batch --limit 20"
