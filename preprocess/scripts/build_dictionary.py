# scripts/build_dictionary.py
"""
raw/*.csv 를 읽어서
- 병해충 / 작물 / 농약 / 품종
각각에 대해 정규화 + 중복제거를 거친
dict_*.json 을 생성.
"""

import pandas as pd
import json
import re
from pathlib import Path


DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def normalize_ko(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # 필요시 한자/특수문자 처리 추가
    return text


def build_pest_dict():
    path = DATA_RAW / "pests_raw.csv"
    df = pd.read_csv(path)

    df["name_ko_norm"] = df["name_ko"].apply(normalize_ko)
    df = df[df["name_ko_norm"] != ""]
    df = df.drop_duplicates(subset=["name_ko_norm"])

    records = []
    for i, row in df.iterrows():
        records.append({
            "id": f"pest_{i:05d}",
            "name_ko": row["name_ko_norm"],
            "latin_name": normalize_ko(row.get("latin_name", "")),
            "host_crops": normalize_ko(row.get("host_crops", "")),
            "category": "pest_or_disease",
            "source": "public_pest_db",
        })

    out_path = DATA_PROCESSED / "dict_pests.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[dict_pests] {len(records)}개 -> {out_path}")


def build_crops_dict():
    path = DATA_RAW / "crops_raw.csv"
    df = pd.read_csv(path)

    df["name_ko_norm"] = df["name_ko"].apply(normalize_ko)
    df = df[df["name_ko_norm"] != ""]
    df = df.drop_duplicates(subset=["name_ko_norm"])

    records = []
    for i, row in df.iterrows():
        records.append({
            "id": f"crop_{i:05d}",
            "name_ko": row["name_ko_norm"],
            "group": normalize_ko(row.get("group", "")),
            "category": "crop",
            "source": "public_crop_db",
        })

    out_path = DATA_PROCESSED / "dict_crops.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[dict_crops] {len(records)}개 -> {out_path}")


def build_pesticides_dict():
    path = DATA_RAW / "pesticides_raw.csv"
    df = pd.read_csv(path)

    df["product_name_norm"] = df["product_name"].apply(normalize_ko)
    df["ingredient_name_norm"] = df["ingredient_name"].apply(normalize_ko)

    df = df[df["product_name_norm"] != ""]
    df = df.drop_duplicates(subset=["product_name_norm"])

    records = []
    for i, row in df.iterrows():
        records.append({
            "id": f"pesticide_{i:05d}",
            "product_name": row["product_name_norm"],
            "ingredient_name": row["ingredient_name_norm"],
            "form": normalize_ko(row.get("form", "")),
            "target_pest": normalize_ko(row.get("target_pest", "")),
            "target_crop": normalize_ko(row.get("target_crop", "")),
            "category": "pesticide",
            "source": "public_pesticide_api",
        })

    out_path = DATA_PROCESSED / "dict_pesticides.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[dict_pesticides] {len(records)}개 -> {out_path}")


def build_varieties_dict():
    path = DATA_RAW / "varieties_raw.csv"
    df = pd.read_csv(path)

    df["variety_name_norm"] = df["variety_name"].apply(normalize_ko)
    df["crop_name_norm"] = df["crop_name"].apply(normalize_ko)

    df = df[df["variety_name_norm"] != ""]
    df = df.drop_duplicates(subset=["variety_name_norm"])

    records = []
    for i, row in df.iterrows():
        records.append({
            "id": f"variety_{i:05d}",
            "variety_name": row["variety_name_norm"],
            "crop_name": row["crop_name_norm"],
            "applicant": normalize_ko(row.get("applicant", "")),
            "category": "variety",
            "source": "public_variety_db",
        })

    out_path = DATA_PROCESSED / "dict_varieties.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[dict_varieties] {len(records)}개 -> {out_path}")


def main():
    build_pest_dict()
    build_crops_dict()
    build_pesticides_dict()
    build_varieties_dict()


if __name__ == "__main__":
    main()
