#!/usr/bin/env bash
set -euo pipefail

log() { echo -e "[$(date +'%H:%M:%S')] $*"; }

# 입력/출력 경로
IN_DIR="${IN_DIR:-/app/org_data}"
HTML_DIR="${HTML_DIR:-/app/html_data}"
JSON_DIR="${JSON_DIR:-/app/data}"
OUT_DIR="${OUT_DIR:-/app/out}"

mkdir -p "$IN_DIR" "$HTML_DIR" "$JSON_DIR" "$OUT_DIR"

shopt -s nullglob
pdfs=("$IN_DIR"/*.pdf)

if [ ${#pdfs[@]} -eq 0 ]; then
  log "❗ ${IN_DIR} 에 PDF가 없습니다. PDF를 넣고 컨테이너를 다시 실행하세요."
  exit 1
fi

############################################
# 1) PDF → HTML
############################################
log "PDF → HTML (recursive, jobs=${PDF_JOBS:-4})"
python /app/pdf_to_html.py \
  --input "$IN_DIR" \
  --output "$HTML_DIR" \
  --recursive \
  --use auto \
  --jobs "${PDF_JOBS:-4}"

############################################
# 2) HTML → JSON(text)  (엔티티 복원 포함)
############################################
log "  HTML → JSON(text)"
mapfile -t htmls < <(find "$HTML_DIR" -type f -name "*.html" | sort)
if [ ${#htmls[@]} -eq 0 ]; then
  log " ${HTML_DIR} 에 HTML이 없습니다."
  exit 1
fi

for html in "${htmls[@]}"; do
  rel="${html#$HTML_DIR/}"
  stem="${rel%.html}"
  out_path="$JSON_DIR/${stem}.json"
  out_dir="$(dirname "$out_path")"
  mkdir -p "$out_dir"

  log "   - $(basename "$html") → $(basename "$out_path")"
  python /app/html_to_text.py --input "$html" --output "$out_path" --as-json
done

############################################
# 3) JSON → CHUNKS (모드 선택)
############################################
log "  JSON → CHUNKS (mode=${CHUNK_MODE:-length})"

# 공통 옵션
COMMON_OPTS=( --input "$JSON_DIR" --output "$OUT_DIR" )
case "${CHUNK_MODE:-length}" in
  length)
    python /app/chunking.py \
      --mode length \
      "${COMMON_OPTS[@]}" \
      --max-tokens "${MAX_TOKENS:-700}" \
      --overlap "${OVERLAP:-100}" \
      --tokenizer-name "${TOKENIZER_NAME:-google/gemma-2-2b}"
    ;;
  semantic)
    python /app/chunking.py \
      --mode semantic \
      "${COMMON_OPTS[@]}" \
      --model-name "${MODEL_NAME:-unsloth/gemma-3-4b-it}" \
      --min-chars "${MIN_CHARS:-50}"
    ;;
  hybrid)
    python /app/chunking.py \
      --mode hybrid \
      "${COMMON_OPTS[@]}" \
      --model-name "${MODEL_NAME:-unsloth/gemma-3-4b-it}" \
      --tokenizer-name "${TOKENIZER_NAME:-google/gemma-2-2b}" \
      --cap-tokens "${CAP_TOKENS:-800}" \
      --overlap "${OVERLAP:-100}"
    ;;
  *)
    log " 지원하지 않는 CHUNK_MODE=${CHUNK_MODE}"
    exit 1
    ;;
esac

log "  파이프라인 완료"
log "   HTML: $HTML_DIR"
log "   JSON: $JSON_DIR"
log "   CHUNKS: $OUT_DIR"
