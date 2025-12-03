# html_to_text.py
import argparse, re, html as html_lib, json
from pathlib import Path
from bs4 import BeautifulSoup

def html_to_plain_text(html_str: str) -> str:
    soup = BeautifulSoup(html_str, "html.parser")

    # 잡음 태그 제거
    for t in soup(["script","style","head","title","meta","link"]):
        t.decompose()

    # 줄바꿈 보존용 치환
    for br in soup.find_all(["br"]):
        br.replace_with("\n")
    for p in soup.find_all(["p","li","tr","div","section"]):
        # 블록 태그는 문단 구분
        if p.text and not p.text.endswith("\n"):
            p.append("\n")

    # 순수 텍스트 추출 + 엔티티 복원
    text = soup.get_text("", strip=False)
    text = html_lib.unescape(text)
    text = text.replace("\xa0", " ")

    # 공백/개행 정리
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def main():
    ap = argparse.ArgumentParser(description="HTML → plain text 또는 JSON(text) 변환")
    ap.add_argument("--input", required=True, help="입력 HTML 파일 경로")
    ap.add_argument("--output", required=True, help="출력 파일 경로 (.txt 또는 .json)")
    ap.add_argument("--as-json", action="store_true", help='JSON({"text": ...}) 형식으로 저장')
    args = ap.parse_args()

    html = Path(args.input).read_text(encoding="utf-8", errors="ignore")
    txt  = html_to_plain_text(html)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --as-json 이거나, 확장자가 .json이면 JSON으로 저장
    if args.as_json or out_path.suffix.lower() == ".json":
        payload = {"text": txt}
        out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    else:
        out_path.write_text(txt, encoding="utf-8")

if __name__ == "__main__":
    main()
