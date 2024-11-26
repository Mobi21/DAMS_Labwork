#%%
import json
import os
import sys
import random
import ssl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import gzip
import chardet
from tqdm import tqdm
from boilerpy3 import extractors

if sys.version_info[0] > 2:
    from http.cookiejar import LWPCookieJar
    from urllib.request import Request, urlopen

else:
    from cookielib import LWPCookieJar
    from urllib2 import Request, urlopen


def load_user_agents(user_agents_file):
    try:
        if user_agents_file.endswith('.gz'):
            with gzip.open(user_agents_file, 'rt') as f:
                return [line.strip() for line in f if line.strip()]
        else:
            with open(user_agents_file) as f:
                return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Failed to load user agents from {user_agents_file}: {e}")
        return None


def load_cookie_jar():
    home_folder = os.getenv('HOME', '.')
    cookie_jar_path = os.path.join(home_folder, '.cookies', 'lwp_cookie_jar')
    os.makedirs(os.path.dirname(cookie_jar_path), exist_ok=True)
    cookie_jar = LWPCookieJar(cookie_jar_path)
    try:
        cookie_jar.load(ignore_discard=True)
    except Exception:
        pass
    return cookie_jar


class WebScraper:
    def __init__(self, user_agents=None):
        self.cookie_jar = load_cookie_jar()
        self.user_agents = user_agents or ['Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0)']
        self.extractors = [
            extractors.CanolaExtractor(),
            extractors.DefaultExtractor(),
            extractors.ArticleExtractor(),
            extractors.LargestContentExtractor(),
            extractors.KeepEverythingExtractor(),
            extractors.NumWordsRulesExtractor(),
            extractors.ArticleSentencesExtractor(),
        ]

    def get_random_user_agent(self):
        return random.choice(self.user_agents)
    def make_request(self, url, verify_ssl=True):
        headers = {'User-Agent': self.get_random_user_agent()}
        if not verify_ssl:
            context = ssl._create_unverified_context()
        else:
            context = None

        request = Request(url, headers=headers)
        self.cookie_jar.add_cookie_header(request)
        try:
            with urlopen(request, context=context) if context else urlopen(request) as response:
                raw_data = response.read()
                encoding = response.headers.get_content_charset() or chardet.detect(raw_data)[
                    'encoding']  # Dynamically detect encoding
                content = raw_data.decode(encoding, errors='replace')  # Use detected encoding to decode content
                self.cookie_jar.extract_cookies(response, request)
                try:
                    self.cookie_jar.save(ignore_discard=True)
                except Exception:
                    pass
            return content
        except Exception:
            print('something wrong!!')
            return content

    @staticmethod
    def requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
        session = requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def extract_text(self, url):
        try:
            html = self.make_request(url).encode().decode('utf-8')
            # First try with boilerpipe extractors
            for extractor in self.extractors:
                content = extractor.get_content(html)
                if content:
                    return content
            # Fallback to BeautifulSoup if no content found
            return self.extract_text_using_bs4(html)
        except Exception as e:
            print(f"Error extracting text from {url}: {e}")
            return None

    @staticmethod
    def extract_text_using_bs4(html):
        soup = BeautifulSoup(html, 'html.parser')
        return ' '.join(soup.stripped_strings)


# Example usage
if __name__ == "__main__":
    scraper = WebScraper()
    with open('google.json', 'r', encoding='utf-8') as file:
        _data = json.load(file)

    # x = _data
    # for i in tqdm(x):
    #     if i["mark"] and i["policy_url"]:
    #         # print(i["manufacturer"])
    #         url = i["policy_url"]
    #         print(url)
    #         res = scraper.extract_text(url)
    #         i['policy_text'] = res

    x = _data
    for i in tqdm(x):
        _list = []
        if i["play_store_policy_link"] is not None:
            print(i["name"])
            url = i["play_store_policy_link"]
            res = scraper.extract_text(url)
            _list.append(res)

        i['policy_text'] = _list

    with open('policies.json', 'w', encoding='utf-8') as file:
        json.dump(x, file, indent=4, ensure_ascii=False)
#%%