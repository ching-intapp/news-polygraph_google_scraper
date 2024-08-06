import os
import json
import requests
import logging
import urllib
import asyncio
import pandas as pd

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
from delphai_scraper_utils.deboilerplating import get_text_from_html
from delphai_scraper_utils.utils import Maybe, get_maybe, remove_duplicates, get_full_title

SCALESERP_KEY = "BCBA058807E94C6B98242339646AF525"
SCRAPER_ID = "google_search"
SIMILARITY_THRESHOLD = 0.6
TOP_N = 15
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

async def call_search_api(query: str):
    params = {
        "api_key": SCALESERP_KEY,
        "q": query,
        "page": 1, # page number
        "num": 30, # results to shows on each page
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
        logging.info("ERROR getting snippets")
        return []
    snippets = []
    for item in response.get("organic_results", []):
        snippet = item.get("snippet")
        title = item.get("title")
        link = item.get("link")
        if snippet:
            # check if the title is cut
            if title and title.endswith("..."):
                full_title = get_full_title(link)
                if full_title:
                    title = full_title
            snippets.append(
                dict(
                    snippet=snippet,
                    link=link,
                    title=title,
                    date=item.get("date"), #
                    domain=item.get("domain"), #
                    query=query.value,
                    src="organic_results",
                )
            )

        for nr in get_maybe(["nested_results"], item, []).value:
            snippet = nr.get("snippet")
            title = item.get("title")
            link = nr.get("link")

            if snippet:
                # check if the title is cut
                if title and title.endswith("..."):
                    full_title = get_full_title(link)
                    if full_title:
                        title = full_title
                snippets.append(
                    dict(
                        snippet=snippet,
                        link=link,
                        title=title,
                        date=item.get("date"), #
                        domain=item.get("domain"), #
                        query=query.value,
                        src="nested_results",
                    )
                )
        snippet = get_maybe(["rich_snippet", "top", "extensions"], item)
        if snippet.success:
            title = item.get("title")
            link = item.get("link")

            # check if the title is cut
            if title and title.endswith("..."):
                full_title = get_full_title(link)
                if full_title:
                    title = full_title
            snippets.append(
                dict(
                    snippet=" ".join(snippet.value),
                    link=link,
                    title=title,
                    date=item.get("date"), #
                    domain=item.get("domain"), #
                    query=query.value,
                    src="rich_snippet/extensions",
                )
            )

        snippet = get_maybe(["rich_snippet", "top", "attributes_flat"], item)
        if snippet.success:
            title = item.get("title")
            link = item.get("link")

             # check if the title is cut
            if title and title.endswith("..."):
                full_title = get_full_title(link)
                if full_title:
                    title = full_title
            snippets.append(
                dict(
                    snippet=item["rich_snippet"]["top"]["attributes_flat"],
                    link=link,
                    title=title,
                    date=item.get("date"), #
                    domain=item.get("domain"), #
                    #title=query.value,
                    query=query.value,
                    src="rich_snippet/attributes_flat",
                )
            )

            for faq_item in get_maybe(["faq"], item, []).value:
                snippet = get_maybe(["answer"], faq_item)
                title = item.get("title")
                link = item.get("link")

                if snippet.success:
                    # check if the title is cut
                    if title and title.endswith("..."):
                        full_title = get_full_title(link)
                        if full_title:
                            title = full_title
                    snippets.append(
                        dict(
                            snippet=snippet.value,
                            link=link,
                            title=title,
                            date=item.get("date"), #
                            domain=item.get("domain"), #
                            #title=query.value,
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
        logging.info(f"[scaleserp] success cache:{url}")
        return api_result.text
    except Exception as ex:
        logging.info(f"[scaleserp] error {repr(ex)}")

# import claim dataset
async def process_claim(claim, label, source, posted, claim_id):
    '''
    output ideal format output file: example at ./output/exp_request_output.json
    add the api search result directly to the json together with the metadata of the claim
    '''
    nltk.data.path.append(os.path.expanduser('~/nltk_data'))

    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(claim, convert_to_tensor=True)
    
    api_result = await call_search_api(claim)
    all_snippets = []
    if api_result:
        try:
            response_json = api_result.json()
        except Exception as ex:
            logging.info(f"Error parsing JSON response: {repr(ex)}")
        else:
            snippets = get_snippets(response_json)
            all_snippets.extend(snippets)
    else:
        logging.info("No results returned from API")

    all_snippets = remove_duplicates(all_snippets)

    for snippet in all_snippets:
        title_embedding = model.encode(snippet["title"], convert_to_tensor=True)
        relevancy = util.pytorch_cos_sim(query_embedding, title_embedding).item()
        snippet["relevancy"] = relevancy

    relevant_snippets = [s for s in all_snippets if s["relevancy"] >= SIMILARITY_THRESHOLD]
    relevant_snippets = sorted(relevant_snippets, key=lambda x: x["relevancy"], reverse=True)[:TOP_N]

    for index, snippet in enumerate(relevant_snippets):
        page_html = ""
        article_text = ""
        try:
            response = await httpx_client.get(snippet["link"], timeout=50)
            page_html = response.text
        except Exception as e:
            page_html = await scaleserp_download(snippet["link"])
        if page_html:
            # delphai ml utils
            extracted_text = get_text_from_html(page_html)
            # trafilatura
            if not extracted_text:
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
                #print(len(extracted_text))
                tokenized_sentences = sent_tokenize(extracted_text)
                snippet["article"] = tokenized_sentences

                #with open(f"article_{index}.html", "w", encoding="utf-8") as file:
                #    file.write("<html><body>")
                #    file.write(extracted_text.replace("\n", "<br>"))
                #    file.write("</body></html>")
                #    logging.info(f"Saved article {index} to article_{index}.html")
            else:
                logging.info(f"No article body found for snippet {index}")

    return {
        "claim": claim,
        "label": label,
        "source": source,
        "posted": posted,
        "claim_id": claim_id,
        "api_result": relevant_snippets
    }

async def main():
    df = pd.read_csv('./claim_dataset.csv')
    claims = df.to_dict(orient='records')[0:3] #set mini size for testing
    testing_query = [
        'Wheaties cereal sticks to magnets because it has metal flakes', 
        'Farmers feed their cattle candy, such as Skittles',
        'The gender of a bell pepper can be identified by counting its lobes',
        ' Kraft Macaroni & Cheese products carry a warning label due to their use of GMO wheat',
        'A photograph shows Barack Obama sitting with Malcolm X and Martin Luther King, Jr.'
        ]
    results = []
    for claim in claims:
        result = await process_claim(
            claim["claim"],
            claim["label"],
            claim["source"],
            claim["posted"],
            claim["claim_id"]
        )
        results.append(result)
    
    with open("./output/exp_request_output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    asyncio.run(main())


# the following output the api search result
# example output file see: ./output/request_output.json
'''
async def main(query):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query, convert_to_tensor=True)
    
   # fetch results from the API
    api_result = await call_search_api(query)
    all_snippets = []
    if api_result:
        try:
            response_json = api_result.json()
        except Exception as ex:
            logging.info(f"Error parsing JSON response: {repr(ex)}")
        else:
            snippets = get_snippets(response_json)
            all_snippets.extend(snippets)
    else:
        logging.info("No results returned from API")

    # remove duplicates and retain the most complete data: having domin, date, title
    all_snippets = remove_duplicates(all_snippets)

    # relevancy scores
    for snippet in all_snippets:
        title_embedding = model.encode(snippet["title"], convert_to_tensor=True)
        relevancy = util.pytorch_cos_sim(query_embedding, title_embedding).item()
        snippet["relevancy"] = relevancy
    relevant_snippets = [s for s in all_snippets if s["relevancy"] >= SIMILARITY_THRESHOLD]
    relevant_snippets = sorted(relevant_snippets, key=lambda x: x["relevancy"], reverse=True)[:TOP_N]

    with open("./output/request_output.json", "w", encoding="utf-8") as f:
        json.dump(relevant_snippets, f, ensure_ascii=False, indent=4)

    # save results to a JSON file
    for index, snippet in enumerate(relevant_snippets):
        page_html = ""
        article_text = ""
        try:
            response = await httpx_client.get(snippet["link"], timeout=50)
            page_html = response.text
        except Exception as e:
            page_html = await scaleserp_download(snippet["link"])
        if page_html:
            # delphai ml utils
            extracted_text = get_text_from_html(page_html)
            # trafilatura
            if not extracted_text:
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
                #print(len(extracted_text))
                tokenized_sentences = sent_tokenize(extracted_text)
                snippet["article"] = tokenized_sentences

                #with open(f"article_{index}.html", "w", encoding="utf-8") as file:
                #    file.write("<html><body>")
                #    file.write(extracted_text.replace("\n", "<br>"))
                #    file.write("</body></html>")
                #    logging.info(f"Saved article {index} to article_{index}.html")
            else:
                logging.info(f"No article body found for snippet {index}")
    with open("./output/request_output.json", "w", encoding="utf-8") as f:
        json.dump(relevant_snippets, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    testing_query = [
        'Wheaties cereal sticks to magnets because it has metal flakes', 
        'Farmers feed their cattle candy, such as Skittles',
        'The gender of a bell pepper can be identified by counting its lobes',
        ' Kraft Macaroni & Cheese products carry a warning label due to their use of GMO wheat',
        'A photograph shows Barack Obama sitting with Malcolm X and Martin Luther King, Jr.'
        ]
    query = testing_query[0]
    asyncio.run(main(query))

'''