from scrap.requests import CrawlRequests
from scrap.extract_document import download_pdf
import logging
import re
import time

logging.basicConfig(level=logging.INFO)

data_params = {
    "menuId": "PS00072",
    "insttFlag": "ATIS",
    "apiFlag": "ATIS_API_FLAG_FARM",
    "currentPageNo": 1
}

def sanitize_filename(title: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "", title)

def main():
    crawl = CrawlRequests()
    all_files = []

    for idx in range(1, 80):  # 1~79 페이지 순회
        logging.info(f"[{idx}] 페이지 크롤링 시작")
        data_params.update({"currentPageNo": idx})  # 여기를 pageIndex가 아니라 currentPageNo로 수정

        try:
            links, titles = crawl.paper_preprocessor(
                url="https://www.nongsaro.go.kr/portal/ps/psz/pszf/apiUnityCall.ps",
                params=data_params,
            )

            if not links or not titles:
                logging.warning(f"[{idx}] 페이지에서 수집된 데이터 없음")
                continue

            for i, (link, title) in enumerate(zip(links, titles), 1):
                safe_title = sanitize_filename(title)
                save_path = f"C:/Users/dm_ohminchan/Model/data/최신영농활용기술/{safe_title}.hwpx"

                success = download_pdf(
                    url=link,
                    save_path=save_path,
                    params=None,  # 링크가 아예 삽입이 되어 있으니 필요하진 않기에 일단 패러미터에서 빼버림
                    timeout=10
                )

                if success:
                    logging.info(f"[{idx}P {i}/{len(links)}] {title} 다운로드 완료")
                    all_files.append(save_path)
                else:
                    logging.warning(f"[{idx}P {i}] {title} 다운로드 실패")

                time.sleep(5)

        except Exception as e:
            logging.error(f"[{idx}] 페이지 처리 중 오류: {e}")
            continue

    return all_files

if __name__ == "__main__":
    results = main()
    logging.info(f"\n총 다운로드 완료 파일 수: {len(results)}")
