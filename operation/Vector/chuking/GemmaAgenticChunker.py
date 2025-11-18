from __future__ import annotations
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os, re
from typing import List

# 전처리: 개행/탭/잡문/캡션 정리
CAPTION_LINE = re.compile(r'(?im)^\s*(표|그림|table|fig(?:ure)?)\s*\d+[^ \n]*.*$', re.M)
PAGE_META_LINE = re.compile(r'(?m)^\s*(쪽|페이지|한국융합학회|저자정보|출처|참고문헌)\s*.*$')

def clean_text_before_chunking(text: str) -> str:
    """
    청킹 전에 텍스트 정규화:
      - 과도한 개행/탭/복수공백 정리
      - 페이지 메타/잡문 제거
      - 단어 중간 개행 복원
      - 표/그림/figure/table 캡션 라인 제거
    """
    text = CAPTION_LINE.sub("", text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[\t ]{2,}', ' ', text)
    text = PAGE_META_LINE.sub("", text)
    text = re.sub(r'(?<=[가-힣a-zA-Z0-9])\n(?=[가-힣a-zA-Z0-9])', ' ', text)
    return text.strip()


class GemmaAgenticChunker:
    def __init__(self, model_name: str = "unsloth/gemma-3-4b-it", max_new_tokens: int = 1024):
        print("Gemma3 모델 로딩 중...")
        self.max_new_tokens = max_new_tokens

        has_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if has_cuda else "cpu")
        supports_bf16 = has_cuda and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if supports_bf16 else (torch.float16 if has_cuda else torch.float32)

        HF_TOKEN = "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"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            torch_dtype=dtype,
            device_map="auto",
        )

        try:
            if has_cuda and getattr(self.model, "device", device).type == "cpu":
                self.model.to("cuda")
        except Exception as e:
            print(f" 강제 GPU 이동 실패(무시 가능): {e}")

        try:
            self.model.config._attn_implementation = "eager"
            if device.type == "cuda":
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(False)
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        print(f"Gemma3 로딩 완료! 장치: {getattr(self.model, 'device', device)} (dtype={dtype})")

    def _build_prompt(self, text: str) -> str:

        messages = [{
            "role": "user",
            "content": f"""
다음 농업 텍스트를 의미 단위로 나누어 주세요.

규칙:
- 각 청크는 하나의 완성된 주제나 개념 포함
- 표나 데이터는 생략
- 청크 수는 내용에 따라 유동적으로 구성
- 다음 형식을 반드시 지킬 것

텍스트:
{text}

형식:
===CHUNK_1===
[내용1]

===CHUNK_2===
[내용2]

===CHUNK_3===
[내용3]
""".strip()
        }]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # 템플릿이 없으면 수동 구성
            return "User:\n" + messages[0]["content"] + "\nAssistant:\n"

    def _truncate_inputs(self, prompt_text: str) -> dict:
        """모델/토크나이저 한계에 맞게 안전하게 토큰 길이 캡 + attention_mask 항상 제공."""
        model_max = getattr(self.tokenizer, "model_max_length", 8192)
        if model_max is None or model_max > 100_000:
            model_max = 8192

        new_tokens = max(1, min(self.max_new_tokens, 1024))
        safety = 256
        max_prompt_tokens = max(512, model_max - new_tokens - safety)

        toks = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        ids = toks.input_ids
        if ids.shape[-1] > max_prompt_tokens:
            ids = ids[:, -max_prompt_tokens:]   # 뒤쪽 보존
        attn = torch.ones_like(ids)
        return {
            "input_ids": ids.to(self.model.device),
            "attention_mask": attn.to(self.model.device),
        }

    # -----------------------------
    # Public API
    # -----------------------------
    def agentic_chunk(self, text: str) -> List[str]:
        # 전처리(개행/탭/잡문/캡션 제거) 먼저
        text = clean_text_before_chunking(text)

        try:
            prompt_text = self._build_prompt(text)
            model_inputs = self._truncate_inputs(prompt_text)

            # 생성 (eager 유지, use_cache on)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max(1, min(self.max_new_tokens, 1024)),
                    temperature=0.2,
                    top_p=0.9,
                    do_sample=True,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            tail = response[-500:] if len(response) > 500 else response
            print("Gemma3 응답(후반부):\n" + tail)
            chunks = self._parse_simple_chunks(response)

            # 파싱 실패 시 폴백: 빈줄 기준 단락 분리(최소 길이 필터)
            if not chunks:
                paras = [p.strip() for p in re.split(r'\n\s*\n', text) if len(p.strip()) >= 50]
                chunks = paras[:]
            return chunks

        except Exception as e:
            print(f"에이전틱 청킹 실패: {e}")
            return []

    def _parse_simple_chunks(self, response: str) -> List[str]:
        """
        LLM 응답에서 청크만 추출:
          - 마커 변형 허용 (===CHUNK_1=== / === CHUNK 1 === / CHUNK 1:)
          - 표/그림 캡션 라인 제거
          - 너무 짧은/너무 긴 조각 길이 보정
        """
        # 1) 정규 마커 패턴 캡처
        matches = re.findall(
            r'===\s*CHUNK[\s_]*(\d+)\s*===\s*(.*?)(?=\n\s*===\s*CHUNK|\Z)',
            response, flags=re.S | re.I
        )
        pieces = [m[1].strip() for m in matches if m[1].strip()]

        # 2) 대안 마커 허용 (CHUNK 1: / = CHUNK 1 = / [CHUNK 1])
        if not pieces:
            matches = re.findall(
                r'(?:^|\n)\s*(?:=+\s*)?CHUNK[\s_]*(\d+)(?:\s*[:=]+)?\s*\n(.*?)(?=\n\s*(?:=+\s*)?CHUNK[\s_]*\d+|\Z)',
                response, flags=re.S | re.I
            )
            pieces = [m[1].strip() for m in matches if m[1].strip()]

        # 3) 캡션/잡문 제거 + 공백 정리
        cleaned = []
        for p in pieces:
            p = CAPTION_LINE.sub("", p)
            p = re.sub(r'[ \t]+', ' ', p)
            p = re.sub(r'\n{3,}', '\n\n', p)
            p = p.strip()
            if p:
                cleaned.append(p)

        # 4) 길이 보정 (짧은 건 버리고, 너무 긴 건 문장경계로 800~1200자 단위 컷)
        out: List[str] = []
        for p in cleaned:
            if len(p) < 50:
                continue
            if len(p) <= 1400:
                out.append(p)
                continue
            # 길면 문장 단위로 800~1200자 권장 길이로 분할
            sents = re.split(r'(?<=[.!?。！？])\s+', p)
            buf = ""
            for s in sents:
                if len(buf) + len(s) <= 1200:
                    buf += (s + " ")
                else:
                    if len(buf) >= 200:
                        out.append(buf.strip())
                    buf = s + " "
            if len(buf) >= 200:
                out.append(buf.strip())

        return out
