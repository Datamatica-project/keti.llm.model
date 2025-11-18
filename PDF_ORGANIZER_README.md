# PDF 정리 스크립트

이 스크립트는 `data/raw` 폴더의 PDF 파일들을 분석하여 구분/학습방식/문서명/수집키워드를 기준으로 정리한 CSV 파일을 생성합니다.

## 사용법

1. 의존성 설치:
```bash
bash install_pdf_deps.sh
```

2. 스크립트 실행:
```bash
python pdf_organizer.py
```

## 출력 파일

- `data/organized_pdfs.csv`: 정리된 PDF 정보가 담긴 CSV 파일

## CSV 컬럼 설명

- **구분**: 문서 유형 (농업, 기술, 교육, 연구, 일반)
- **학습방식**: ML 학습 방식 (지도학습, 비지도학습, 강화학습)
- **문서명**: PDF 파일명
- **수집키워드**: 문서에서 추출한 주요 키워드 (최대 5개)

## 분류 기준

### 구분 분류
- 농업: 파일명에 '농업', '농사', '작물', '재배' 포함
- 기술: 파일명에 '기술', 'tech', '개발' 포함
- 교육: 파일명에 '교육', '학습', 'education' 포함
- 연구: 파일명에 '연구', 'research', '논문' 포함
- 일반: 위 조건에 해당하지 않는 경우

### 학습방식 분류
- 지도학습: 내용에 'classification', '분류', 'supervised' 포함
- 비지도학습: 내용에 'clustering', '군집', 'unsupervised' 포함
- 강화학습: 내용에 'reinforcement', '강화학습', 'reward' 포함
