from scrap.requests import CrawlRequests
from scrap.extract_document import download_pdf
import time
import logging
import re

logging.basicConfig(level=logging.INFO)

data_params = {
    "menuId": "PS00199",
    "pageIndex": 1,
    "pageSize": 10
}

def sanitize_filename(title):
    return re.sub(r'[\\/*?:"<>|]', "", title)

def main():
    crawl = CrawlRequests()
    all_files = []

    for idx in range(1, 80):  # 1페이지부터 79페이지까지 반복
        logging.info(f"페이지 {idx} 크롤링 시작")
        data_params.update({"pageIndex": idx})

        # 매 페이지마다 최신 파라미터로 데이터 수집
        params, titles = crawl.form_preprocessor(
            url="https://www.nongsaro.go.kr/portal/ps/psz/psza/contentMain.ps",
            params=data_params
        )

        if not params or not titles:
            logging.warning(f"페이지 {idx} 수집된 데이터 없음")
            continue

        for i, (param, title) in enumerate(zip(params, titles), 1):
            safe_title = sanitize_filename(title)
            save_path = f"C:/Users/dm_ohminchan/Model/data/raw/{safe_title}.pdf"

            try:
                download_pdf(
                    url="https://www.nongsaro.go.kr/portal/contentsFileDownload.do",
                    save_path=save_path,
                    params=param,
                    timeout=10
                )
                logging.info(f"{idx}P {i}/{len(params)} {title} 다운로드 완료")
                all_files.append(save_path)
            except Exception as e:
                logging.error(f"{idx}P {i} {title} 다운로드 실패: {e}")

            time.sleep(5)

    return all_files

if __name__ == "__main__":
    print(main())
