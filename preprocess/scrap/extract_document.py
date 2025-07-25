import requests
import os
import logging

def download_pdf(
    url: str,
    save_path: str,
    params: dict = None,
    timeout: int = 10
) -> bool:
    try:

        headers  = {
            "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers, params=params, timeout=timeout)
        response.raise_for_status()

        # 저장 디렉토리 없으면 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "wb") as f:
            f.write(response.content)

        logging.info(f"Downloaded: {save_path}")
        return True

    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")
        return False
