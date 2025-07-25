import logging
from typing import Dict, Optional, Any, List, Tuple
import requests
from lxml import html
import re

class CrawlRequests:

    def __init__(self, headers: Optional[Dict[str, str]] = None, timeout: int =10):
        self.session = requests.Session()
        self.headers = headers or {
            "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
        }

        self.timeout = timeout


    def get_data(self, url: str, params: Optional[Dict[str, Any]] = None)->  Optional[requests.Response]:
        try:
            response = self.session.get(url, headers=self.headers, timeout=self.timeout, params=params)
            response.raise_for_status()
            logging.info(f"get요청 성공 {response.url}")
            return response
        except Exception as e:
            logging.error(f"get 요청 실패 사유 : {e}")
            return None

    def convert_html_form(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[html.HtmlElement]:
        response = self.get_data(url, params)
        if response is None:
            return None

        try:
            tree = html.fromstring(response.content)
            logging.info(f"HTML 파싱 성공: {url}")
            return tree
        except Exception as e:
            logging.error(f"HTML 파싱 실패: {url} | 오류: {e}")
            return None

    def form_preprocessor(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Tuple[List,List]]:
        response = self.convert_html_form(url, params)
        if response is None:
            return None

        data_lst = []
        title_lst = []

        files_tree = response.xpath('//a[1][contains(@onclick, "fncFileDown")]/@onclick')
        titles = response.xpath('//td[@class="txt-l"]/text()')


        try:
            for file, title in zip(files_tree, titles):
                match = re.search(r":fncFileDown\(\s*'([^']+)'\s*,\s*'[^']+'\s*,\s*'([^']+)'", file)
                if match:
                    result = {
                        "cntntsNo": match.group(1),
                        "fileSeCode": match.group(2),
                        "fileSn" : "3"
                    }

                    title_lst.append(title)
                    data_lst.append(result)

            return data_lst, title_lst

        except Exception as e:
            logging.error(f"HTML 파싱 실패: {url} | 오류: {e}")
            return None

    def paper_preprocessor(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[
        Tuple[List[str], List[str]]]:
        response = self.convert_html_form(url, params)
        if response is None:
            return None

        file_tags = response.xpath('//a[@class="ico02"]')
        title_tags = response.xpath('//td[@class="tl bT_subject"]//a/text()')
        file_links = []
        titles = []

        for tag, title in zip(file_tags, title_tags):
            onclick = tag.xpath('./@onclick')[0]
            match = re.search(r"'(http[^']+)'", onclick)
            if match:
                raw_url = match.group(1)
                https_url = re.sub(r"^http:", "https:", raw_url)
                file_links.append(https_url)
                titles.append(title.strip())

        return file_links, titles



