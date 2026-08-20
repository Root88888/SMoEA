#!/usr/bin/env bash
# =============================================================================
# scripts/publish_artifacts.sh — 驗證通過後，一次完成標記與上傳
#
#   bash scripts/publish_artifacts.sh <org>/<repo> <驗證報告目錄> <暫存目錄> \
#       <producer runs 根目錄> [方法…]
#
# 依序做三件事，任一步失敗即停止：
#   1. 檢查每個方法的等價驗證報告都是 PASS（bitwise_identical=true）
#   2. 以內容定址的 run_id 重新標記（權重用 hardlink，不佔額外空間）
#   3. 上傳到 Hugging Face repo（push_artifact.py 預設公開，--private 轉私有）
#
# producer 的參考版本以 <runs 根目錄>/<方法>/<run_id>/prepare/merged_model 尋找。
# 同一個方法底下有多個 run 時停下來要求指定，不會自己挑一個——這與
# fetch_artifact.py 的規矩一致。
#
# 前置：hf auth login（或設定 HF_TOKEN）。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

USAGE="用法：bash scripts/publish_artifacts.sh <org>/<repo> <報告目錄> <暫存目錄> <producer runs 根目錄> [方法…]"
REPO_ID="${1:?$USAGE}"
VERIFY_DIR="${2:?$USAGE}"
STAGE_DIR="${3:?$USAGE}"
RUNS_ROOT="${4:?$USAGE}"
shift 4
METHODS="${*:-ta ties_only dare_ties_ta}"

# producer 的參考版本由目錄結構找出來，不寫死 run_id：同一個方法有多個 run
# 時停下來要求指定，不自己挑一個（與 fetch_artifact.py 同一條規矩）。
declare -A PROD=()
for M in $METHODS; do
    MATCHES=()
    while IFS= read -r D; do MATCHES+=("$D"); done < <(
        find "$RUNS_ROOT/$M" -mindepth 3 -maxdepth 3 -type d \
             -path "*/prepare/merged_model" 2>/dev/null | sort)
    case ${#MATCHES[@]} in
      0) echo "✗ $RUNS_ROOT/$M 底下找不到 prepare/merged_model"; exit 1 ;;
      1) PROD[$M]="${MATCHES[0]}" ;;
      *) echo "✗ $M 有多個 run，請把 <runs 根目錄> 指到只含一個的位置："
         printf '     %s\n' "${MATCHES[@]}"; exit 1 ;;
    esac
done

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
        --out-root "$STAGE_DIR"
done

echo
echo "== 3/3 上傳到 $REPO_ID =="
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
