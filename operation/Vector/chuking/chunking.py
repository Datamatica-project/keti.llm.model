#!/usr/bin/env python3
"""
독립 실행형 청킹 파이프라인 (기존 노트북/클래스를 수정하지 않음)

- 세 가지 모드: 의미 기반(semantic), 길이 기반(length), 하이브리드(hybrid)
- 입력(txt/json/md/html)을 정규화된 청크 JSON으로 저장
- JSON은 깊은 구조도 재귀적으로 평탄화하여 본문을 최대한 보존
- 긴 본문은 6,000자/400자 오버랩 윈도우로 나눠 LLM을 여러 번 호출(semantic/hybrid)
- hybrid: 의미 청킹 후 토큰 기반 캡으로 2차 세분화

출력 스키마(참조 파일 호환):
{chunk_id, index, documents, date, content}
"""

from __future__ import annotations

from GemmaAgenticChunker import GemmaAgenticChunker
import argparse
import os
import re
import json
import uuid
import glob
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Iterable, Optional

from transformers import AutoTokenizer
from dotenv import load_dotenv  # ✅ 추가

try:
    import boto3
except Exception:
    boto3 = None

# .env 로드 (.env가 현재 작업 디렉토리/상위 디렉토리에 있을 때 자동으로 찾아줌)
load_dotenv()

JSON_TEXT_KEYS = {
    "text", "content", "body", "paragraph", "section", "abstract",
    "introduction", "conclusion", "description"
}
JSON_TITLE_KEYS = {"title", "heading", "section_title", "chapter", "document"}


def _flatten_json_text(obj) -> str:
    parts: List[str] = []

    def walk(x):
        if x is None:
            return
        if isinstance(x, str):
            s = x.strip()
            if s:
                parts.append(s)
            return
        if isinstance(x, (int, float)):
            parts.append(str(x))
            return
        if isinstance(x, list):
            for it in x:
                walk(it)
            return
        if isinstance(x, dict):
            # 우선 title/text 후보 키들을 먼저 본다
            for k in list(JSON_TITLE_KEYS | JSON_TEXT_KEYS):
                if k in x:
                    walk(x[k])
            # 나머지 키들 순회
            for k, v in x.items():
                if k in JSON_TITLE_KEYS or k in JSON_TEXT_KEYS:
                    continue
                walk(v)
            return

    walk(obj)

    txt = "\n\n".join(p for p in parts if p)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = re.sub(r"[ \t]{2,}", " ", txt)
    txt = re.sub(
        r'(?<=[가-힣A-Za-z0-9])\n(?=[가-힣A-Za-z0-9])',
        ' ',
    )
    return txt.strip()


def _split_long_text_windows(
    text: str,
    target_chars: int = 6000,
    overlap_chars: int = 400,
) -> List[str]:
    if len(text) <= target_chars:
        return [text]

    chunks: List[str] = []
    i = 0
    while i < len(text):
        end = min(i + target_chars, len(text))
        chunks.append(text[i:end])
        if end == len(text):
            break
        i = max(end - overlap_chars, i + 1)
    return chunks


def read_text_from_path(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()

    if path.lower().endswith(".json"):
        try:
            return _flatten_json_text(json.loads(data))
        except Exception:
            return data
    return data


def iter_input_files(input_path: str) -> Iterable[str]:
    if os.path.isdir(input_path):
        for ext in ("*.txt", "*.json", "*.md", "*.html"):
            for p in glob.glob(os.path.join(input_path, ext)):
                yield p
    else:
        yield input_path


@dataclass
class LengthCfg:
    max_tokens: int = 700
    overlap: int = 100
    tokenizer_name: str = "google/gemma-2-2b"


def ensure_tokenizer(name: str):
    """환경변수 HF_TOKEN을 사용해서 토크나이저 로드 (하드코딩 토큰 제거)."""
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN 환경변수가 설정되어 있지 않습니다. "
            ".env 파일이나 시스템 환경변수를 확인하세요."
        )

    tok = AutoTokenizer.from_pretrained(name, token=hf_token)

    if not getattr(tok, "eos_token_id", None):
        # eos_token이 없을 경우 sep/pad 등에서 적당히 대체
        tok.eos_token = tok.eos_token or tok.sep_token or tok.pad_token or "</s>"

    return tok


def split_by_tokens(text: str, cfg: LengthCfg) -> List[str]:
    tok = ensure_tokenizer(cfg.tokenizer_name)
    units = re.split(r"(\n\s*\n|[.!?]\s+)", text)

    chunks: List[str] = []
    buf = ""

    for i in range(0, len(units), 2):
        part = units[i]
        delim = units[i + 1] if i + 1 < len(units) else ""
        candidate = (buf + part + delim).strip()
        ids = tok(candidate, add_special_tokens=False).input_ids

        if len(ids) <= cfg.max_tokens:
            buf = candidate + " "
        else:
            if buf.strip():
                chunks.append(buf.strip())

            ids = tok(part, add_special_tokens=False).input_ids
            start = 0
            while start < len(ids):
                end = min(start + cfg.max_tokens, len(ids))
                piece = tok.decode(ids[start:end])
                chunks.append(piece.strip())
                start = max(end - cfg.overlap, end)

            buf = delim.strip() + " "

    if buf.strip():
        chunks.append(buf.strip())

    if cfg.overlap > 0 and chunks:
        out: List[str] = []
        for i, ch in enumerate(chunks):
            if i == 0:
                out.append(ch)
                continue
            prev = chunks[i - 1]
            prev_ids = tok(prev, add_special_tokens=False).input_ids
            tail_ids = (
                prev_ids[-cfg.overlap:]
                if len(prev_ids) > cfg.overlap
                else prev_ids
            )
            prefix = tok.decode(tail_ids)
            out.append((prefix + " " + ch).strip())
        chunks = out

    return [c for c in (s.strip() for s in chunks) if c]


@dataclass
class SemanticCfg:
    model_name: str = "unsloth/gemma-3-4b-it"
    min_chars: int = 50


def run_semantic_chunk(text: str, cfg: SemanticCfg) -> List[str]:
    chunker = GemmaAgenticChunker(model_name=cfg.model_name)
    chunks = chunker.agentic_chunk(text)
    return [
        c
        for c in chunks
        if isinstance(c, str) and len(c.strip()) >= cfg.min_chars
    ]


def cap_chunks_by_tokens(chunks: List[str], cfg: LengthCfg) -> List[str]:
    tok = ensure_tokenizer(cfg.tokenizer_name)
    capped: List[str] = []

    for ch in chunks:
        ids = tok(ch, add_special_tokens=False).input_ids
        if len(ids) <= cfg.max_tokens:
            capped.append(ch.strip())
            continue

        start = 0
        while start < len(ids):
            end = min(start + cfg.max_tokens, len(ids))
            piece = tok.decode(ids[start:end])
            capped.append(piece.strip())
            start = max(end - cfg.overlap, end)

    return [c for c in capped if c]


def normalize_chunks_for_storage(
    chunks: List[str],
    document_name: str,
) -> List[Dict]:
    today = dt.date.today().isoformat()
    out: List[Dict] = []

    for i, ch in enumerate(chunks):
        out.append(
            {
                "chunk_id": str(uuid.uuid4()),
                "index": i,
                "documents": document_name,
                "date": today,
                "content": ch,
            }
        )
    return out


def save_json(data: List[Dict], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def maybe_upload_minio(
    local_path: str,
    bucket: str,
    key: str,
    endpoint_url: str,
    access_key: str,
    secret_key: str,
):
    if boto3 is None:
        print("boto3 미설치 → MinIO 업로드 생략")
        return

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    with open(local_path, "rb") as f:
        s3.put_object(Bucket=bucket, Key=key, Body=f.read())
    print(f"업로드 완료: s3://{bucket}/{key} @ {endpoint_url}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="독립 실행형 청킹 파이프라인 (비침입형)")
    p.add_argument("--mode", choices=["length", "semantic", "hybrid"], default="length")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--doc-name", default=None)

    p.add_argument("--max-tokens", type=int, default=700)
    p.add_argument("--overlap", type=int, default=100)
    p.add_argument("--tokenizer-name", default="google/gemma-2-2b")

    p.add_argument("--model-name", default="unsloth/gemma-3-4b-it")
    p.add_argument("--min-chars", type=int, default=50)

    p.add_argument("--cap-tokens", type=int, default=800)

    p.add_argument("--minio-bucket", default=None)
    p.add_argument("--minio-prefix", default="data/")
    p.add_argument("--minio-endpoint", default="http://localhost:9000")
    p.add_argument("--minio-access", default="minio")
    p.add_argument("--minio-secret", default="miniostorage")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    length_cfg = LengthCfg(
        max_tokens=args.max_tokens,
        overlap=args.overlap,
        tokenizer_name=args.tokenizer_name,
    )
    semantic_cfg = SemanticCfg(
        model_name=args.model_name,
        min_chars=args.min_chars,
    )

    os.makedirs(args.output, exist_ok=True)

    for path in iter_input_files(args.input):
        raw_text = read_text_from_path(path)
        segments = _split_long_text_windows(
            raw_text,
            target_chars=6000,
            overlap_chars=400,
        )

        base = os.path.splitext(os.path.basename(path))[0]
        doc_name = args.doc-name if args.doc_name else base
        mode = args.mode

        chunks: List[str] = []

        if mode == "length":
            chunks = split_by_tokens(raw_text, length_cfg)
        elif mode == "semantic":
            all_sem: List[str] = []
            for seg in segments:
                all_sem.extend(run_semantic_chunk(seg, semantic_cfg))
            chunks = all_sem
        elif mode == "hybrid":
            all_sem: List[str] = []
            for seg in segments:
                all_sem.extend(run_semantic_chunk(seg, semantic_cfg))
            cap_cfg = LengthCfg(
                max_tokens=args.cap_tokens,
                overlap=args.overlap,
                tokenizer_name=args.tokenizer_name,
            )
            chunks = cap_chunks_by_tokens(all_sem, cap_cfg)

        payload = normalize_chunks_for_storage(chunks, document_name=doc_name)
        out_path = os.path.join(args.output, f"{base}_chunks.json")
        save_json(payload, out_path)
        print(f"{len(payload)}개의 청크 생성 완료 -> {out_path}")

        if args.minio_bucket:
            key = f"{args.minio_prefix.rstrip('/')}/{os.path.basename(out_path)}"
            maybe_upload_minio(
                out_path,
                args.minio_bucket,
                key,
                args.minio_endpoint,
                args.minio_access,
                args.minio_secret,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
