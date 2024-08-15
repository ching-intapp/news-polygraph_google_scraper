
import logging
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import Any, List, Dict, Union, Tuple

# helper functions
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
    #if snippet.get("title"):
    #    completeness += 1
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

def get_full_title(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title_tag = soup.find('title')
            if title_tag:
                return title_tag.get_text()
    except Exception as e:
        logging.info(f"Error fetching full title from {url}: {e}")
    return None
