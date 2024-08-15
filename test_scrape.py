import requests
import httpx
import urllib
import logging
import json
import asyncio
import os

import trafilatura #

from httpx import AsyncClient
from delphai_scraper_utils import ScraperClient

from newspaper import fulltext #
from newspaper import Article


'''
SCALESERP_KEY = 'API-KEY'
SCRAPER_ID = "google_search"
httpx_client = ScraperClient(scraper_id=SCRAPER_ID)

params = {
  'api_key': 'BCBA058807E94C6B98242339646AF525',
  'q': 'A photograph showing a group of astronauts without their helmets on indicates that the moon landing was staged.',
  'page': '1',
  'num': '10'
}

api_result = requests.get('https://api.scaleserp.com/search', params)
print('api_result:', api_result.json())
with open("request_output.json", "w", encoding="utf-8") as f:
    json.dump(api_result.json(), f, ensure_ascii=False, indent=4)
'''

def get_config(key: str, default=None):
    return os.getenv(key, default)

async def scaleserp_download(url: str):
    encoded_url = urllib.parse.quote(url, safe="")
    print(f"Encoded URL: {encoded_url}")
    params = {
        "api_key": SCALESERP_KEY,
        "q": f"cache:{url}",
        "engine": "google",
        "output": "html",
    }
    print(f"Params: {params}")
    try:
        api_result = await httpx_client.get(
            "https://api.scaleserp.com/search", params=params, timeout=50
        )
        logging.info(f"[scaleserp] success cache:{url}") #logger
        print(f"API result: {api_result}")
        return api_result.text
    except Exception as ex:
        logging.info(f"[scaleserp] error {repr(ex)}") #logger
        print(f"Exception: {ex}")
        return None

async def main_test():
    example_link = "https://www.yahoo.com/news/fact-check-theres-more-image-221200761.html"
    #response = await httpx_client.get(example_link, timeout=50)
    full_content = await scaleserp_download(example_link)
    #print(response.text)
    if full_content:
        print("Full content retrieved successfully:")
        print(full_content)
    else:
        print("Failed to retrieve content")

#if __name__ == "__main__":
#    asyncio.run(main_test())


test_url = 'https://www.yahoo.com/news/fact-check-theres-more-image-221200761.html' #html parser
#html = requests.get(test_url).text
#text = fulltext(html)
#print('news:', text)

article = Article(test_url).text
print('newspaper:', article)

downloaded = trafilatura.fetch_url(test_url)
text = trafilatura.extract(downloaded)
print('trafilatura:', text)