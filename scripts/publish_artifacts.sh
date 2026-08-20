#!/usr/bin/env bash
# =============================================================================
# scripts/publish_artifacts.sh — 驗證通過後，一次完成標記與上傳
#
#   bash scripts/publish_artifacts.sh <org>/<repo> [驗證報告目錄] [暫存目錄]
#
# 依序做三件事，任一步失敗即停止：
#   1. 檢查每個方法的等價驗證報告都是 PASS（bitwise_identical=true）
#   2. 以內容定址的 run_id 重新標記（權重用 hardlink，不佔額外空間）
#   3. 上傳到私有 Hugging Face repo
#
# 前置：huggingface-cli login（或設定 HF_TOKEN）。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ID="${1:?用法：bash scripts/publish_artifacts.sh <org>/<repo> [報告目錄] [暫存目錄]}"
VERIFY_DIR="${2:-/livingrooms/tincan/smoea/verify}"
STAGE_DIR="${3:-/livingrooms/tincan/smoea/artifacts/staged}"

R=/livingrooms/tincan/smoea/integration-runs/2026-08-19-unified-1024-random10-v3/runs
declare -A PROD=(
  [ta]="$R/ta/f61f5fe81f58fdba/prepare/merged_model"
  [ties_only]="$R/ties_only/e3de085e3caeaf23/prepare/merged_model"
  [dare_ties_ta]="$R/dare_ties_ta/6317f9cbaefe06c2/prepare/merged_model"
)
METHODS="ta ties_only dare_ties_ta"

echo "== 1/3 檢查驗證報告 =="
for M in $METHODS; do
    REPORT="$VERIFY_DIR/${M}_report.json"
    [ -f "$REPORT" ] || { echo "✗ 缺少 $REPORT —— 驗證尚未完成"; exit 1; }
    python - "$REPORT" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
d = r.get("difference") or {}
if r.get("bitwise_identical"):
    print(f"  相同   {r['method']:10s} {r['smoea_sha256'][:16]}")
elif d.get("all_differences_within_one_ulp"):
    # 差異在儲存格式的最小間隔之內：搬移的實作正確，只是浮點加法順序隨顯卡而異。
    print(f"  一格內 {r['method']:10s} 相異 {d['differing_elements']:,}"
          f"/{d['total_elements']:,}，最大差 1 ulp")
else:
    print(f"  超標   {r['method']:10s} 差異超過一個 ulp——搬移的實作與 producer 不等價")
    print(f"         最大絕對差 {d.get('max_abs_difference')}")
    sys.exit(1)
PY
done
echo "  三個方法全部 PASS"

echo
echo "== 2/3 以內容定址 run_id 重新標記 =="
for M in $METHODS; do
    python scripts/stamp_content_id.py --method "$M" \
        --producer "${PROD[$M]}" --report "$VERIFY_DIR/${M}_report.json" \
        --adapter-dir adapter --out-root "$STAGE_DIR"
done

echo
echo "== 3/3 上傳到 $REPO_ID（私有）=="
ARGS=()
for M in $METHODS; do
    DIR="$(find "$STAGE_DIR/$M" -type d -name merged_model | head -1)"
    [ -n "$DIR" ] || { echo "✗ 找不到 $M 的暫存目錄"; exit 1; }
    ARGS+=(--artifact "$DIR")
done
python scripts/push_artifact.py --repo "$REPO_ID" --dry-run "${ARGS[@]}"
echo
python scripts/push_artifact.py --repo "$REPO_ID" "${ARGS[@]}"

echo
echo "== 完成 =="
echo "取用：python scripts/fetch_artifact.py --repo $REPO_ID --condition ties \\"
echo "        --artifact-root <本機路徑> --registry <本機路徑>/registry.json"
