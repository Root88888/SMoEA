#!/usr/bin/env bash
# =============================================================================
# scripts/setup_workspace.sh — 本機一鍵前置（Ubuntu ＋ NVIDIA GPU ＋ conda）
#
# 用途：clone 本 repo、把「需自行準備」的檔案放好之後，在 repo 根目錄執行
# 本腳本；結束時 main.py 即可直接使用（例如測試 system/rejection.py）。
# 可重複執行：已完成的步驟自動跳過。router 的離線準備全部由本腳本代辦。
#
#   bash scripts/setup_workspace.sh                     # 不準備 artifact（拒絕走 base）
#   bash scripts/setup_workspace.sh --artifacts merge   # 以本機 adapter 線上合成
#   bash scripts/setup_workspace.sh --artifacts fetch --hf-repo <org>/<repo>
#
# ─────────────────────────────────────────────────────────────────────────────
# 【需自行準備】（腳本會逐項檢查，缺哪項會明講）
#   1. conda（miniconda 即可）與 NVIDIA 驅動已安裝
#   2. Router 訓練樣本 → dataset/train_data/task{N}_train.json
#      benchmark 才需要 dataset/test_data/，線上服務建置不要求。
#   3. adapters → adapter/task{N}/（自 Google Drive 下載解壓後攤平放入；
#        每個任務一個目錄，內含 adapter 檔或 checkpoint-*/）
#   （單位說明書 assets/unit_descriptions.json 隨 repo 自帶，無需準備）
#
# 【腳本代辦】conda env 建置（於專案內 .conda/smoea，與既有環境完全
# 隔離）＋鎖定依賴、環境體檢、查詢嵌入計算
# （首次自動下載嵌入模型 ~1.3GB；GPU 數分鐘）、路由資產建置，
# 以及（選用）rejection artifact 的準備。
# base model 與裁決模型（各 ~16GB）會在首次執行 main.py 時自動下載。
#
# 【rejection artifact】--artifacts 決定拒絕分支有哪些選項可用：
#   （不指定）  只有 base。之後仍可隨時單獨跑 merge/fetch 腳本補上
#   merge      以本機 adapter/ 線上合成 ta、ties、dare-ties
#              （每份約 3.76 GB、需 GPU；已存在者自動跳過）
#   fetch      自 --hf-repo 指定的 Hugging Face repo 取得現成 artifact
# 準備好的項目會登記進 <artifact-root>/registry.json，執行期以
# `:rejection use <id>`（互動）或 `--artifact <id>`（批次）選用。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

ARTIFACT_MODE=""
ARTIFACT_ROOT="$(pwd)/artifacts"   # 與 configs 的預設一致，開箱即用
HF_REPO=""
ARTIFACT_METHODS="ta ties dare-ties"
while [ $# -gt 0 ]; do
    case "$1" in
        --artifacts)      ARTIFACT_MODE="$2"; shift 2 ;;
        --artifact-root)  ARTIFACT_ROOT="$2"; shift 2 ;;
        --hf-repo)        HF_REPO="$2"; shift 2 ;;
        --methods)        ARTIFACT_METHODS="${2//,/ }"; shift 2 ;;
        *) echo "未知參數：$1"; exit 2 ;;
    esac
done
case "$ARTIFACT_MODE" in
    ""|merge|fetch) ;;
    *) echo "--artifacts 只接受 merge 或 fetch"; exit 2 ;;
esac
[ "$ARTIFACT_MODE" = "fetch" ] && [ -z "$HF_REPO" ] \
  && { echo "--artifacts fetch 需要 --hf-repo <org>/<repo>"; exit 2; }

echo "== SMoEA workspace setup @ $(pwd) =="

fail() { echo "✗ $1"; echo "  → $2"; exit 1; }

# ---- 0/5 檢查「需自行準備」清單 ----
[ "$(ls dataset/train_data/task*.json* 2>/dev/null | wc -l)" -ge 1 ] \
  || fail "dataset/train_data/ 沒有任務樣本"
[ -f assets/unit_descriptions.json ] \
  || fail "缺 assets/unit_descriptions.json（repo 自帶）"
[ "$(ls -d adapter/task* 2>/dev/null | wc -l)" -ge 1 ] \
  || fail "adapter/ 沒有任務目錄" "自 Google Drive"
echo "[0/5] 需自行準備的檔案齊全"
echo "      Router 訓練樣本 $(ls dataset/train_data | wc -l) 檔、adapter $(ls -d adapter/task* | wc -l) 個任務"

# ---- 1/5 conda env ----
# 已 activate 某個環境（非 base）→ 直接使用、不另建；
# 否則尋找/建置具名 env「smoea」。conda 不在 PATH 時自動到常見
# 安裝位置尋找（部分容器/機器的 shell 不會自動初始化 conda）。
ensure_deps() {
    python -c "import numpy, scipy, sklearn, torch, transformers, peft" 2>/dev/null \
      || { echo "      依賴不全，安裝鎖定依賴（含 torch 下載，首次 10-20 分鐘）…"; \
           python -m pip install -r requirements-lock-twcc.txt; }
}
act() {   # conda activate/deactivate 與 set -u 的相容包裝
    set +u; conda "$@"; set -u
}
if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" != "base" ]; then
    echo "[1/5] 使用當前已啟用的環境：${CONDA_DEFAULT_ENV}"
else
    if ! command -v conda >/dev/null 2>&1; then
        for c in ~/miniconda3 ~/anaconda3 /opt/conda /opt/miniconda3; do
            [ -f "$c/etc/profile.d/conda.sh" ] && source "$c/etc/profile.d/conda.sh" && break
        done
    fi
    command -v conda >/dev/null 2>&1 \
      || fail "找不到 conda" "安裝 miniconda；或先手動 activate 你的環境再重跑本腳本"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    while [ -n "${CONDA_DEFAULT_ENV:-}" ]; do act deactivate; done
    # 環境建在專案資料夾內（.conda/smoea）：與使用者既有環境完全隔離、
    # 不撞名；刪除專案目錄即完整移除
    ENV_DIR="$(pwd)/.conda/smoea"
    if [ ! -x "$ENV_DIR/bin/python" ]; then
        echo "[1/5] 於專案內建置 conda env（$ENV_DIR）…"
        conda create -y -p "$ENV_DIR" python=3.12
    else
        echo "[1/5] 專案內 conda env 已存在（$ENV_DIR）"
    fi
    act activate "$ENV_DIR"
fi
export PYTHONNOUSERSITE=1
ensure_deps
# 把防護固化進 env：之後單獨 conda activate 也自帶（README 步驟 4 的前提）
conda env config vars set PYTHONNOUSERSITE=1 >/dev/null 2>&1 || true

# ---- 2/5 環境體檢 ----
echo "[2/5] 環境體檢"
python scripts/check_env.py || fail "體檢未過" "照上方 FAIL 提示處置後重跑本腳本"

# ---- 3/5 路由資產（缺則建置；含查詢嵌入計算，快取後不重算）----
mkdir -p results
if [ ! -f assets/router_assets.npz ]; then
    echo "[3/5] 建置路由資產（首次含嵌入計算；GPU 數分鐘）…"
    python scripts/build_router_assets.py --serving-only
else
    echo "[3/5] 路由資產已存在，跳過（要重建：先刪 assets/router_assets.npz）"
fi

# ---- 4/5 rejection artifact（選用）----
REGISTRY="$ARTIFACT_ROOT/registry.json"
if [ -z "$ARTIFACT_MODE" ]; then
    echo "[4/5] 未指定 --artifacts，拒絕分支只會有 base"
    echo "      之後要補：bash scripts/setup_workspace.sh --artifacts merge"
else
    mkdir -p "$ARTIFACT_ROOT"
    echo "[4/5] 準備 rejection artifact（$ARTIFACT_MODE）→ $ARTIFACT_ROOT"
    if [ "$ARTIFACT_MODE" = "merge" ]; then
        # 先擋掉池不完整的情況：合成要跑數十分鐘，不該做到一半才失敗。
        N_ADAPTER="$(ls -d adapter/task* 2>/dev/null | wc -l)"
        [ "$N_ADAPTER" -eq 150 ] \
          || fail "線上合成需要完整的 pool150，目前只有 $N_ADAPTER 個 adapter" \
                  "補齊 adapter/task{N}/ 後重跑，或改用 --artifacts fetch"
    fi
    for METHOD in $ARTIFACT_METHODS; do
        if [ "$ARTIFACT_MODE" = "merge" ]; then
            python scripts/merge_pool150.py --method "$METHOD" \
                --adapter-dir adapter --artifact-root "$ARTIFACT_ROOT" \
                --registry "$REGISTRY" --register-as "$METHOD" \
                --set system.dtype=bfloat16 \
              || fail "合成 $METHOD 失敗" "照上方錯誤處置後重跑（已完成者會自動跳過）"
        else
            python scripts/fetch_artifact.py --repo "$HF_REPO" \
                --condition "$METHOD" --artifact-root "$ARTIFACT_ROOT" \
                --registry "$REGISTRY" --register-as "$METHOD" \
              || fail "取得 $METHOD 失敗" "確認 repo 內容與 HF_TOKEN 後重跑"
        fi
    done
    echo "      registry：$REGISTRY"
fi

# ---- 5/5 完成 ----
echo
echo "== 全部就緒 =="
echo "之後每次開終端機（在任何目錄皆可執行）："
echo "  conda activate ${CONDA_PREFIX}"
echo "測試（互動模式；輸入域外 query 會進 system/rejection.py；"
echo "首次執行會自動下載 base model 與裁決模型，各 ~16GB）："
echo "  python main.py --mode interactive"
echo "批次（拒絕樣本輸出落在 results/main_batch_outputs.jsonl）："
echo "  python main.py --mode batch --limit 20"
if [ -n "$ARTIFACT_MODE" ]; then
    echo
    echo "拒絕分支要用準備好的 artifact，加上 registry 設定："
    echo "  python main.py --mode interactive --set system.artifact_registry=$REGISTRY"
    echo "互動中 :rejection list 看可選項目、:rejection use <id> 切換；"
    echo "批次用 --artifact <id> 指定整批共用哪一個。"
fi
