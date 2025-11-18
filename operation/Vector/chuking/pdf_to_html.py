"""
PDF → HTML 일괄 변환기 (폴더 전체 변환 가능)

- 하나의 PDF 또는 폴더 내 모든 PDF를 변환 (--recursive로 하위 폴더 포함 가능)
- pdf2htmlEX가 있으면 우선 사용, 없거나 실패 시 PyMuPDF(fitz)로 폴백
- (--preserve-dirs 사용 시) 입력 폴더 구조를 출력에도 보존
- 변환 결과 요약 index.html 생성

사용 예:
  python pdf_to_html.py --input org_data/ --output html_data/
  python pdf_to_html.py --input a.pdf --output html_data/
  python pdf_to_html.py --input org_data/ --output html_data/ --recursive --jobs 4 --preserve-dirs
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import html
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple



# PDF 탐색

def find_pdfs(input_path: Path, recursive: bool) -> List[Path]:
    """입력 경로에서 PDF 파일을 모두 탐색"""
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]
    if input_path.is_dir():
        it = input_path.rglob("*.pdf") if recursive else input_path.glob("*.pdf")
        return [p for p in it if p.is_file()]
    return []



# 출력 경로 계산 (폴더 구조 보존 옵션)

def out_path_for(
    pdf_path: Path,
    out_dir: Path,
    base_dir: Path,
    preserve_dirs: bool,
) -> Path:
    """
    출력 HTML 경로 생성.
    - preserve_dirs=True면 base_dir 기준 상대경로를 out_dir에 그대로 재현
    - base_dir 밖 파일이 들어오면 그냥 평면 저장(안전장치)
    """
    if preserve_dirs:
        try:
            rel = pdf_path.relative_to(base_dir)  # base_dir 기준 상대경로
            dest = out_dir / rel.with_suffix(".html")
        except ValueError:
            dest = out_dir / (pdf_path.stem + ".html")
    else:
        dest = out_dir / (pdf_path.stem + ".html")

    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest

# 변환기 구현

def have_pdf2htmlex() -> bool:
    return shutil.which("pdf2htmlEX") is not None


def run_pdf2htmlex(
    pdf: Path,
    html_out: Path,
    first_page: Optional[int],
    last_page: Optional[int],
    embed: bool,
    zoom: float,
    extra_args: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """pdf2htmlEX를 이용한 변환 수행"""
    cmd = [
        "pdf2htmlEX",
        "--zoom", str(zoom),
        "--process-outline", "1",
        "--embed-css", "1" if embed else "0",
        "--embed-font", "1" if embed else "0",
        "--embed-image", "1" if embed else "0",
        "--embed-javascript", "1" if embed else "0",
    ]
    if first_page:
        cmd += ["--first-page", str(first_page)]
    if last_page:
        cmd += ["--last-page", str(last_page)]
    if extra_args:
        cmd += extra_args
    cmd += [str(pdf), str(html_out)]

    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        ok = res.returncode == 0 and html_out.exists()
        return ok, (res.stdout or ("pdf2htmlEX 성공" if ok else "pdf2htmlEX 실패"))
    except Exception as e:
        return False, f"pdf2htmlEX 실행 오류: {e}"


def run_pymupdf(
    pdf: Path,
    html_out: Path,
    first_page: Optional[int],
    last_page: Optional[int],
    zoom: float,  # PyMuPDF의 get_text('html')에는 직접 확대 없음
) -> Tuple[bool, str]:
    """PyMuPDF로 변환 (pdf2htmlEX 미사용 시)"""
    try:
        import fitz  # PyMuPDF
    except Exception:
        return False, "PyMuPDF(fitz)가 설치되어 있지 않습니다. pip install pymupdf 필요"

    try:
        doc = fitz.open(str(pdf))
        total = doc.page_count
        start = max(1, first_page) if first_page else 1
        end = min(total, last_page) if last_page else total
        parts: List[str] = []

        for pno in range(start - 1, end):
            page = doc.load_page(pno)
            try:
                parts.append(page.get_text("html"))
            except Exception:
                # HTML 추출 실패 시 텍스트라도 보존
                parts.append("<pre>" + html.escape(page.get_text("text") or "") + "</pre>")

        doc.close()
        html_doc = (
            f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{html.escape(pdf.stem)}</title>"
            "<style>body{font-family:sans-serif;line-height:1.5} .page{margin:1.5rem 0}</style></head><body>"
        )
        for i, h in enumerate(parts, 1):
            html_doc += f"<section class='page' id='p{i}'>" + h + "</section>\n"
        html_doc += "</body></html>"

        html_out.write_text(html_doc, encoding="utf-8")
        return True, f"PyMuPDF 변환 완료: {end - start + 1}페이지"
    except Exception as e:
        return False, f"PyMuPDF 변환 오류: {e}"


# 단일 파일 변환

def convert_one(
    pdf: Path,
    out_dir: Path,
    method: str,
    recursive: bool,
    embed: bool,
    zoom: float,
    first_page: Optional[int],
    last_page: Optional[int],
    extra_args: Optional[List[str]],
    base_dir: Path,
    preserve_dirs: bool,
) -> Tuple[Path, bool, str]:
    out_html = out_path_for(pdf, out_dir, base_dir, preserve_dirs)

    if method == "pdf2htmlex" or (method == "auto" and have_pdf2htmlex()):
        ok, msg = run_pdf2htmlex(pdf, out_html, first_page, last_page, embed, zoom, extra_args)
        if ok:
            return out_html, True, msg
        ok2, msg2 = run_pymupdf(pdf, out_html, first_page, last_page, zoom)
        return out_html, ok2, (msg + " | 폴백: " + msg2)
    else:
        ok, msg = run_pymupdf(pdf, out_html, first_page, last_page, zoom)
        return out_html, ok, msg


# index.html 생성

def write_index(out_dir: Path, entries: List[Tuple[Path, bool, str]]):
    rows = []
    for html_path, ok, msg in entries:
        # out_dir 기준의 상대경로를 링크에 사용 (하위 폴더 보존 시에도 정상 링크)
        rel = html_path.relative_to(out_dir)
        name = html.escape(str(rel))
        status = "✅" if ok else "❌"
        rows.append(
            f"<tr><td>{status}</td><td><a href='{name}'>{name}</a></td><td><pre>{html.escape(msg)}</pre></td></tr>"
        )
    page = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>PDF 변환 결과 목록</title>"
        "<style>body{font-family:sans-serif} table{border-collapse:collapse;width:100%} td,th{border:1px solid #ddd;padding:8px} pre{white-space:pre-wrap}</style></head><body>"
        "<h1>PDF → HTML 변환 결과</h1>"
        f"<p>총 {len(entries)}개 파일</p>"
        "<table><thead><tr><th>상태</th><th>파일명</th><th>로그</th></tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table></body></html>"
    )
    (out_dir / "index.html").write_text(page, encoding="utf-8")


# CLI 파서 및 메인

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PDF 파일을 HTML로 일괄 변환 (pdf2htmlEX 우선, PyMuPDF 폴백)")
    p.add_argument("--input", required=True, help="입력 PDF 파일 또는 폴더 경로 (예: org_data/)")
    p.add_argument("--output", required=True, help="출력 HTML 폴더 경로 (예: html_data/)")
    p.add_argument("--recursive", action="store_true", help="하위 폴더까지 탐색")
    p.add_argument("--use", choices=["auto", "pdf2htmlex", "pymupdf"], default="auto", help="사용할 변환 방식")
    p.add_argument("--embed", action="store_true", help="(pdf2htmlEX) CSS/폰트/이미지 통합 임베딩")
    p.add_argument("--zoom", type=float, default=1.0, help="확대 배율 (pdf2htmlEX용)")
    p.add_argument("--first-page", type=int, default=None, help="시작 페이지")
    p.add_argument("--last-page", type=int, default=None, help="마지막 페이지")
    p.add_argument("--jobs", type=int, default=1, help="병렬 처리 개수")
    p.add_argument("--no-index", action="store_true", help="index.html 생성을 생략")
    p.add_argument("--preserve-dirs", action="store_true", help="입력 폴더 구조를 출력에도 그대로 보존")
    p.add_argument("--pdf2htmlex-args", nargs=argparse.REMAINDER, help="pdf2htmlEX 추가 인자 (-- 이후 작성)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    in_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = find_pdfs(in_path, args.recursive)
    if not pdfs:
        print(f"[경고] PDF 파일이 없습니다: {in_path}")
        return 0

    print(f"총 {len(pdfs)}개 PDF 발견. 출력 폴더: {out_dir}")

    base_dir = in_path if in_path.is_dir() else in_path.parent
    jobs = max(1, int(args.jobs))
    entries: List[Tuple[Path, bool, str]] = []

    def _task(pdf: Path) -> Tuple[Path, bool, str]:
        html_out, ok, msg = convert_one(
            pdf=pdf,
            out_dir=out_dir,
            method=args.use,
            recursive=args.recursive,
            embed=bool(args.embed),
            zoom=float(args.zoom),
            first_page=args.first_page,
            last_page=args.last_page,
            extra_args=(args.pdf2htmlex_args or None),
            base_dir=base_dir,
            preserve_dirs=args.preserve_dirs,
        )
        status = "성공" if ok else "실패"
        print(f"[{status}] {pdf} → {html_out.relative_to(out_dir)}")
        if not ok:
            print(f"  ↳ {msg}")
        return html_out, ok, msg

    if jobs == 1:
        for pdf in pdfs:
            entries.append(_task(pdf))
    else:
        with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(_task, pdf) for pdf in pdfs]
            for fu in cf.as_completed(futs):
                entries.append(fu.result())

    if not args.no_index:
        write_index(out_dir, entries)
        print(f"index.html 생성 완료: {out_dir / 'index.html'}")

    ok_count = sum(1 for _, ok, _ in entries if ok)
    print(f"변환 완료. 성공: {ok_count}/{len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
