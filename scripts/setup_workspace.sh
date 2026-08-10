#!/usr/bin/env bash
# =============================================================================
# scripts/setup_workspace.sh — 國網一鍵前置：環境 → 檔案 → 路由資產 → 可測試
#
# 用途：clone 本 repo 後在 repo 根目錄執行本腳本，結束時 main.py 即可直接
# 使用（例如測試 system/rejection.py）。腳本可重複執行：已完成的步驟自動
# 跳過。你不需要理解 router 內部——本腳本把它的離線準備全部代辦。
#
#   bash scripts/setup_workspace.sh
#
# 完成後的測試指令（腳本結尾也會再印一次）：
#   source /work/specproj4/router_repo_test/env.sh
#   python main.py --mode interactive
# =============================================================================
set -euo pipefail

# ---- 國網固定路徑（環境不同時只需改這一區）----
ENV_PREFIX=/work/specproj4/envs/smoea
ENV_SH=/work/specproj4/router_repo_test/env.sh
DATA_SRC=/work/specproj4/SMOEA/MoEA-Trainer/dataset/natural_instructions/data/selected_10_tasks
CACHE_SRC=/work/specproj4/SMOEA/MoEA-Trainer/_cache/task_hierarchy_bge-large-en-v1.5_input
ADAPTER_SRC=/work/specproj4/router_repo_test/SMoEA/adapter
VERIFIER_MODEL=/work/specproj4/SMOEA/models/Llama-3.1-8B-Instruct

cd "$(dirname "$0")/.."
REPO_DIR=$(pwd)
echo "== SMoEA workspace setup @ $REPO_DIR =="

# ---- 1/5 環境（conda env 不存在則建置並安裝鎖定依賴）----
source "$(conda info --base)/etc/profile.d/conda.sh"
while [ -n "${CONDA_DEFAULT_ENV:-}" ]; do conda deactivate; done
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    echo "[1/5] 建置 conda env（首次約 10-20 分鐘）…"
    conda create -y -p "$ENV_PREFIX" python=3.12
    conda activate "$ENV_PREFIX"
    export PYTHONNOUSERSITE=1
    python -m pip install -r requirements-lock-twcc.txt
else
    echo "[1/5] conda env 已存在，跳過建置"
    conda activate "$ENV_PREFIX"
    export PYTHONNOUSERSITE=1
fi
export HF_HOME=/work/specproj4/hf_cache
export PIP_CACHE_DIR=/work/specproj4/pip_cache
if [ ! -f "$ENV_SH" ]; then
    cat > "$ENV_SH" <<EOF
while [ -n "\$CONDA_DEFAULT_ENV" ]; do conda deactivate; done
conda activate $ENV_PREFIX
export PYTHONNOUSERSITE=1
export HF_HOME=/work/specproj4/hf_cache
export PIP_CACHE_DIR=/work/specproj4/pip_cache
EOF
    echo "      已建立 $ENV_SH（之後每次登入：source 它即可）"
fi
python scripts/check_env.py

# ---- 2/5 任務樣本 ----
mkdir -p dataset/train_data dataset/test_data assets results
if [ "$(ls dataset/train_data/task*.json* 2>/dev/null | wc -l)" -lt 50 ]; then
    echo "[2/5] 複製任務樣本…"
    cp "$DATA_SRC"/train_data/task*.json* dataset/train_data/
    cp "$DATA_SRC"/test_data/task*.json*  dataset/test_data/
else
    echo "[2/5] 任務樣本已就位，跳過"
fi
echo "      train $(ls dataset/train_data | wc -l) 檔 / test $(ls dataset/test_data | wc -l) 檔（預期 50 / 65）"

# ---- 3/5 嵌入快取與單位說明書 ----
if [ "$(ls assets/emb_task*.npz 2>/dev/null | wc -l)" -lt 50 ]; then
    echo "[3/5] 複製嵌入快取…"
    cp "$CACHE_SRC"/emb_task*.npz      assets/
    cp "$CACHE_SRC"/emb_test_task*.npz assets/
else
    echo "[3/5] 嵌入快取已就位，跳過"
fi
[ -f assets/unit_descriptions.json ] || cp "$CACHE_SRC"/v7_unit_descriptions_50task.json assets/unit_descriptions.json
echo "      嵌入 $(ls assets/emb_task*.npz | wc -l)+$(ls assets/emb_test_task*.npz | wc -l) 檔（預期 50+65）、說明書 OK"

# ---- 4/5 adapters（缺時以符號連結掛現成權重，不佔空間）----
mkdir -p adapter
if [ -z "$(ls -A adapter 2>/dev/null)" ] && [ -d "$ADAPTER_SRC" ] \
   && [ "$ADAPTER_SRC" != "$REPO_DIR/adapter" ]; then
    echo "[4/5] 連結 adapters ← $ADAPTER_SRC"
    ln -s "$ADAPTER_SRC"/task* adapter/
fi
echo "      adapter $(ls adapter 2>/dev/null | wc -l) 個任務目錄"

# ---- 5/5 路由資產（缺則離線建置，純 CPU 數分鐘）----
if [ ! -f assets/router_assets.npz ]; then
    echo "[5/5] 建置路由資產…"
    python scripts/build_router_assets.py
else
    echo "[5/5] 路由資產已存在，跳過（要重建：先刪 assets/router_assets.npz）"
fi

echo
echo "== 全部就緒 =="
echo "之後每次登入："
echo "  source $ENV_SH"
echo "測試（互動模式；輸入域外 query 會進 system/rejection.py）："
echo "  python main.py --mode interactive --set verifier.model_path=$VERIFIER_MODEL"
echo "批次（拒絕樣本輸出落在 results/main_batch_outputs.jsonl）："
echo "  python main.py --mode batch --limit 20"
