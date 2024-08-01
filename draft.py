import os
import json
import requests
import logging
import urllib
import asyncio

import trafilatura
from newspaper import Article

import nltk
from nltk.tokenize import sent_tokenize

from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import Any, List, Dict, Union, Tuple
from delphai_scraper_utils import ScraperClient

SCALESERP_KEY = "BCBA058807E94C6B98242339646AF525" #str(get_config("SCALESERP_KEY"))
SCRAPER_ID = "google_search"
SIMILARITY_THRESHOLD = 0.5 #float #float(str(get_config("SIMILARITY_THRESHOLD")))
httpx_client = ScraperClient(scraper_id=SCRAPER_ID)

#os.environ['NLTK_DATA'] = '/Users/ycyang/nltk_data/tokenizers/punkt'
os.environ['NLTK_DATA'] = os.path.expanduser('~/nltk_data')

DEFAULT_CRAWL_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
}

#helper functions
@dataclass
class Maybe:
    value: Any
    success: bool
    message: str

def get_maybe(
    path: List[Any],
    tree: Union[List[Any], Tuple[Any], Dict[Any, Any]],
    default_value=None,
) -> Maybe:
    """
    Extract branch from tree using path
    """
    if tree is None:
        return Maybe(value=default_value, success=False, message="tree is None")
    res = tree
    pos = []
    for p in path:
        pos.append(p)
        try:
            res = res[p]
        except (TypeError, IndexError, KeyError) as ex:
            return Maybe(
                value=default_value, success=False, message=f"{pos} {repr(ex)}"
            )
    if res is None:
        return Maybe(value=default_value, success=False, message="res is None")
    return Maybe(value=res, success=True, message="")

def calculate_completeness(snippet):
    completeness = 0
    if snippet.get("date"):
        completeness += 1
    if snippet.get("domain"):
        completeness += 1
    if snippet.get("title"):
        completeness += 1
    if snippet.get("link"):
        completeness += 1
    return completeness

def remove_duplicates(snippets):
    unique_snippets = {}
    for snippet in snippets:
        key = (snippet["link"], snippet["title"])
        if key in unique_snippets:
            existing_snippet = unique_snippets[key]
            if calculate_completeness(snippet) > calculate_completeness(existing_snippet):
                unique_snippets[key] = snippet
        else:
            unique_snippets[key] = snippet
    return list(unique_snippets.values())

#---------------------------------------------------------#

async def call_search_api(query: str, page: int):
    params = {
        "api_key": SCALESERP_KEY,
        "q": query,
        "page": page,
        "num": 10, #set 30 for example
    }
    api_result = requests.get('https://api.scaleserp.com/search', params)
    #try:
    #    api_result = await httpx_client.get(
    #        "https://api.scaleserp.com/search", params=params
    #    )
    #    logging.info(f"search for {query}") #logger
    #    return api_result
    #except Exception as ex:
    #    logging.info(f"[httpx_error] {repr(ex)}") #logger
    return api_result

def get_snippets(response):
    query = get_maybe(["search_parameters", "q"], response)
    if not query.success:
        logging.info("ERROR getting snippets") #logger
        return []

    snippets = []

    for item in response.get("organic_results", []):
        snippet = item.get("snippet")
        if snippet:
            snippets.append(
                dict(
                    snippet=snippet,
                    link=item.get("link"),
                    title=item.get("title"),
                    date=item.get("date"), #
                    domain=item.get("domain"), #
                    query=query.value,
                    src="organic_results",
                )
            )

        for nr in get_maybe(["nested_results"], item, []).value:
            snippet = nr.get("snippet")
            if snippet:
                snippets.append(
                    dict(
                        snippet=snippet,
                        link=nr.get("link"),
                        title=item.get("title"),
                        date=item.get("date"), #
                        domain=item.get("domain"), #
                        query=query.value,
                        src="nested_results",
                    )
                )
        snippet = get_maybe(["rich_snippet", "top", "extensions"], item)
        if snippet.success:
            snippets.append(
                dict(
                    snippet=" ".join(snippet.value),
                    link=item.get("link"),
                    title=item.get("title"),
                    date=item.get("date"), #
                    domain=item.get("domain"), #
                    query=query.value,
                    src="rich_snippet/extensions",
                )
            )

        snippet = get_maybe(["rich_snippet", "top", "attributes_flat"], item)
        if snippet.success:
            snippets.append(
                dict(
                    snippet=item["rich_snippet"]["top"]["attributes_flat"],
                    link=item.get("link"),
                    date=item.get("date"), #
                    domain=item.get("domain"), #
                    title=query.value,
                    query=query.value,
                    src="rich_snippet/attributes_flat",
                )
            )

            for faq_item in get_maybe(["faq"], item, []).value:
                snippet = get_maybe(["answer"], faq_item)
                if snippet.success:
                    snippets.append(
                        dict(
                            snippet=snippet.value,
                            link=item.get("link"),
                            date=item.get("date"), #
                            domain=item.get("domain"), #
                            title=query.value,
                            query=query.value,
                            src="faq",
                        )
                    )
    return snippets

async def scaleserp_download(url: str):
    encoded_url = urllib.parse.quote(url, safe="")
    params = {
        "api_key": SCALESERP_KEY,
        "q": f"cache:{url}",
        "engine": "google",
        "output": "html",
    }
    try:
        api_result = await httpx_client.get(
            "https://api.scaleserp.com/search", params=params, timeout=50
        )
        logging.info(f"[scaleserp] success cache:{url}") #logger
        return api_result.text
    except Exception as ex:
        logging.info(f"[scaleserp] error {repr(ex)}") #logger


#-------------------------------------------------------------------------------#

async def main(query):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    all_snippets = []
    # !filtering
    for page in range(1, 3): #do one call and set more result to be returned (parameter:num=30)
        api_result = await call_search_api(query, page)
        if api_result:
            try:
                response_json = api_result.json()
            except Exception as ex:
                logging.info(f"Error parsing JSON response: {repr(ex)}")
                continue
            snippets = get_snippets(response_json)
            all_snippets.extend(snippets)
        else:
            logging.info(f"No results for page {page}")

    # helper function: remove duplicates and keep completeness
    all_snippets = remove_duplicates(all_snippets)

    # relevancy scores
    for snippet in all_snippets:
        title_embedding = model.encode(snippet["title"], convert_to_tensor=True)
        relevancy = util.pytorch_cos_sim(query_embedding, title_embedding).item()
        snippet["relevancy"] = relevancy
    relevant_snippets = [s for s in all_snippets if s["relevancy"] >= SIMILARITY_THRESHOLD]
    relevant_snippets = sorted(relevant_snippets, key=lambda x: x["relevancy"], reverse=True)[:10]

    with open("request_output.json", "w", encoding="utf-8") as f:
        json.dump(relevant_snippets, f, ensure_ascii=False, indent=4)

    # Save results to a JSON file
    for index, snippet in enumerate(relevant_snippets):
        page_html = ""
        article_text = ""
        try:
            response = await httpx_client.get(snippet["link"], timeout=50)
            page_html = response.text
        except Exception as e:
            page_html = await scaleserp_download(snippet["link"])

        if page_html:
            # trafilatura
            extracted_text = trafilatura.extract(page_html)
            if not extracted_text:
                # newspaper3k
                article = Article(snippet["link"])
                article.set_html(page_html)
                article.parse()
                extracted_text = article.text
            if not extracted_text:
                # BeautifulSoup
                soup = BeautifulSoup(page_html, 'html.parser')
                article_body = soup.find('body')
                if article_body:
                    paragraphs = article_body.find_all('p')
                    extracted_text = "\n".join([p.get_text() for p in paragraphs])
            if extracted_text: 
                print(len(extracted_text))
                tokenized_sentences = sent_tokenize(extracted_text)
                snippet["article"] = tokenized_sentences

                #with open(f"article_{index}.html", "w", encoding="utf-8") as file:
                #    file.write("<html><body>")
                #    file.write(extracted_text.replace("\n", "<br>"))
                #    file.write("</body></html>")
                #    logging.info(f"Saved article {index} to article_{index}.html")
            else:
                logging.info(f"No article body found for snippet {index}")
    with open("request_output.json", "w", encoding="utf-8") as f:
        json.dump(relevant_snippets, f, ensure_ascii=False, indent=4) #page_range:1-2

'''
async def main(query):
    api_result = await call_search_api(query)
    snippets = get_snippets(api_result.json())
    #with open("request_output.json", "w", encoding="utf-8") as f:
    #    json.dump(snippets, f, ensure_ascii=False, indent=4)
    #print(snippets, type(snippets))
    for snippet in snippets:
        page_html = ""
        try:
            response = await httpx_client.get(snippet["link"], timeout=50)
            page_html = response.text
            print(page_html)
        except Exception as e:
            page_html = await scaleserp_download(snippet["link"])
            print(page_html)
        
        # apply page_html
'''

if __name__ == "__main__":
    testing_query = [
        'Wheaties cereal sticks to magnets because it has metal flakes', 
        'Farmers feed their cattle candy, such as Skittles',
        'The gender of a bell pepper can be identified by counting its lobes',
        ' Kraft Macaroni & Cheese products carry a warning label due to their use of GMO wheat',
        'A photograph shows Barack Obama sitting with Malcolm X and Martin Luther King, Jr.'
        ]
    query = testing_query[-1]
    asyncio.run(main(query))

