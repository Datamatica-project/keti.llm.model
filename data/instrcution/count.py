import json, sys, pathlib, glob

def load_items(path: pathlib.Path):
    text = path.read_text(encoding="utf-8").strip()
    # JSONL?
    if "\n" in text and text.lstrip().split("\n", 1)[0].rstrip().endswith("}"):
        items = []
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except json.JSONDecodeError as e:
                print(f"[WARN] {path.name}:{i} JSONL 파싱 실패: {e}", file=sys.stderr)
        return items
    # JSON 배열?
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # dict 안에 리스트가 있는 경우도 지원 (e.g., {"data":[...]})
            for v in data.values():
                if isinstance(v, list):
                    return v
            return [data]
    except json.JSONDecodeError as e:
        print(f"[WARN] {path.name} JSON 파싱 실패: {e}", file=sys.stderr)
    return []

def is_valid_qa(obj):
    return isinstance(obj, dict) and obj.get("QUESTION") and obj.get("ANSWER")

def main(patterns):
    files = []
    for p in patterns:
        files.extend([pathlib.Path(x) for x in glob.glob(p, recursive=True)])
    if not files:
        print("대상 파일이 없습니다. 예) python count_qa.py data/*.json", file=sys.stderr)
        sys.exit(1)

    total = 0
    valid = 0
    by_persp = {}
    by_source = {}

    for f in files:
        items = load_items(f)
        total += len(items)
        v = sum(1 for x in items if is_valid_qa(x))
        valid += v
        # breakdowns
        for x in items:
            if is_valid_qa(x):
                by_persp[x.get("perspective","<none>")] = by_persp.get(x.get("perspective","<none>"), 0) + 1
                by_source[x.get("source","<none>")] = by_source.get(x.get("source","<none>"), 0) + 1

    print("=== Synthetic QA 개수 집계 ===")
    print(f"- 로드된 항목(원본 그대로): {total}")
    print(f"- 유효한 QA(QUESTION & ANSWER 존재): {valid}")
    print("\n[관점별(perspective) 분포]")
    for k, v in sorted(by_persp.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k}: {v}")
    print("\n[소스별(source) 분포]")
    for k, v in sorted(by_source.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k}: {v}")

if __name__ == "__main__":
    # 사용법:
    #   python count_qa.py data/*.json
    #   python count_qa.py data/**/*.json
    #   python count_qa.py dataset.jsonl
    if len(sys.argv) < 2:
        print("사용법: python count_qa.py <파일/패턴...>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1:])
