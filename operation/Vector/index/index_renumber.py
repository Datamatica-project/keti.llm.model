import json

# 입력 JSON 파일 경로
input_path = "metadata.json"
output_path = "metadata.json"

# JSON 로드
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# index 오름차순 정렬 (혹시 섞여 있을 경우 대비)
data = sorted(data, key=lambda x: x.get("index", 0))

# index를 0부터 다시 재부여
for new_idx, item in enumerate(data):
    item["index"] = new_idx

# 저장
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✔ index 재정렬 완료!")
