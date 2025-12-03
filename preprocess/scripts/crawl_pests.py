# scripts/crawl_pests.py
"""
병해·병해충 이름/학명/기주작물 등을 크롤링해서
data/raw/pests_raw.csv 로 저장하는 스크립트.
실제 URL / CSS 셀렉터는 사이트 구조에 맞게 수정해야 함.
"""

import requests
from bs4 import BeautifulSoup
import csv
from time import sleep

LIST_URL = "https://example.com/pests/list"      # TODO: 실제 목록 URL
DETAIL_BASE = "https://example.com"              # TODO: 상세 URL prefix

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def fetch_pest_list():
    """
    병해충 목록 페이지에서 이름 + 상세 링크 수집
    """
    res = requests.get(LIST_URL, headers=HEADERS)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

    pests = []
    # TODO: 아래 CSS 선택자는 실제 구조 보고 수정
    for row in soup.select("table.pest-list tbody tr"):
        name_ko = row.select_one("td.name").get_text(strip=True)
        detail_href = row.select_one("td.name a")["href"]

        pests.append({
            "name_ko": name_ko,
            "detail_url": DETAIL_BASE + detail_href
        })

    return pests


def enrich_with_details(pests):
    """
    각 병해충 상세 페이지에서 학명/기주작물 등 추가 수집
    """
    enriched = []
    for p in pests:
        try:
            res = requests.get(p["detail_url"], headers=HEADERS)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")

            # TODO: 실제 페이지 구조에 맞게 셀렉터 수정
            latin_el = soup.select_one("span.latin-name")
            host_el = soup.select_one("div.host-crops")

            p["latin_name"] = latin_el.get_text(strip=True) if latin_el else ""
            p["host_crops"] = host_el.get_text(strip=True) if host_el else ""
        except Exception as e:
            print("Error on", p["detail_url"], e)
            p.setdefault("latin_name", "")
            p.setdefault("host_crops", "")

        enriched.append(p)
        sleep(0.2)  # 과도한 요청 방지

    return enriched


def main():
    pests = fetch_pest_list()
    pests = enrich_with_details(pests)

    out_path = "data/raw/pests_raw.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name_ko", "latin_name", "host_crops", "detail_url"],
        )
        writer.writeheader()
        writer.writerows(pests)

    print(f"[병해충] {len(pests)}개 저장 -> {out_path}")


if __name__ == "__main__":
    main()
