#!/usr/bin/env python3
"""
PDF 파일을 구분/학습방식/문서명/수집키워드 기준으로 정리하여 CSV로 생성하는 스크립트
"""

import os
import csv
import PyPDF2
from pathlib import Path
import re
from typing import List, Dict, Tuple

class PDFOrganizer:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.raw_path = self.data_path / "raw"
        self.output_path = self.data_path / "organized_pdfs.csv"
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """PDF에서 텍스트 추출"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            print(f"PDF 읽기 오류 {pdf_path}: {e}")
            return ""
    
    def classify_document(self, filename: str, content: str) -> Dict[str, str]:
        """문서 분류 및 메타데이터 추출"""
        # 기본값 설정
        classification = {
            "구분": "일반",
            "학습방식": "지도학습",
            "문서명": filename,
            "수집키워드": ""
        }
        
        # 파일명 기반 분류
        filename_lower = filename.lower()
        
        # 구분 분류
        if any(keyword in filename_lower for keyword in ['농업', '농사', '작물', '재배']):
            classification["구분"] = "농업"
        elif any(keyword in filename_lower for keyword in ['기술', 'tech', '개발']):
            classification["구분"] = "기술"
        elif any(keyword in filename_lower for keyword in ['교육', '학습', 'education']):
            classification["구분"] = "교육"
        elif any(keyword in filename_lower for keyword in ['연구', 'research', '논문']):
            classification["구분"] = "연구"
        
        # 학습방식 분류
        if any(keyword in content.lower() for keyword in ['classification', '분류', 'supervised']):
            classification["학습방식"] = "지도학습"
        elif any(keyword in content.lower() for keyword in ['clustering', '군집', 'unsupervised']):
            classification["학습방식"] = "비지도학습"
        elif any(keyword in content.lower() for keyword in ['reinforcement', '강화학습', 'reward']):
            classification["학습방식"] = "강화학습"
        
        # 수집키워드 추출 (내용에서 주요 키워드 추출)
        keywords = self.extract_keywords(content)
        classification["수집키워드"] = ", ".join(keywords[:5])  # 상위 5개 키워드
        
        return classification
    
    def extract_keywords(self, content: str) -> List[str]:
        """텍스트에서 키워드 추출"""
        # 한글, 영문 단어 추출
        words = re.findall(r'[가-힣]{2,}|[a-zA-Z]{3,}', content)
        
        # 불용어 제거
        stopwords = {'그리고', '하지만', '그러나', '또한', '이것', '그것', '이런', '그런', 
                    'and', 'but', 'the', 'is', 'are', 'was', 'were', 'this', 'that'}
        
        keywords = [word for word in words if word.lower() not in stopwords]
        
        # 빈도 계산 및 상위 키워드 반환
        from collections import Counter
        word_freq = Counter(keywords)
        
        return [word for word, count in word_freq.most_common(10)]
    
    def process_all_pdfs(self) -> List[Dict[str, str]]:
        """모든 PDF 파일 처리"""
        results = []
        
        if not self.raw_path.exists():
            print(f"경로가 존재하지 않습니다: {self.raw_path}")
            return results
        
        pdf_files = list(self.raw_path.glob("*.pdf")) + list(self.raw_path.glob("*.PDF"))
        
        for pdf_file in pdf_files:
            print(f"처리 중: {pdf_file.name}")
            
            # PDF 내용 추출
            content = self.extract_text_from_pdf(pdf_file)
            
            # 문서 분류
            classification = self.classify_document(pdf_file.name, content)
            
            results.append(classification)
        
        return results
    
    def save_to_csv(self, data: List[Dict[str, str]]) -> None:
        """CSV 파일로 저장"""
        if not data:
            print("저장할 데이터가 없습니다.")
            return
        
        fieldnames = ["구분", "학습방식", "문서명", "수집키워드"]
        
        with open(self.output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"CSV 파일 생성 완료: {self.output_path}")
    
    def run(self) -> None:
        """전체 프로세스 실행"""
        print("PDF 파일 정리 시작...")
        
        # PDF 파일들 처리
        processed_data = self.process_all_pdfs()
        
        # CSV로 저장
        self.save_to_csv(processed_data)
        
        print(f"총 {len(processed_data)}개 파일 처리 완료")

def main():
    # 데이터 경로 설정
    data_path = "./data"
    
    # PDF 정리기 실행
    organizer = PDFOrganizer(data_path)
    organizer.run()

if __name__ == "__main__":
    main()
