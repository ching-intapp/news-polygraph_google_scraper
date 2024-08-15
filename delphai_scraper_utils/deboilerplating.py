import trafilatura
from goose3 import Goose
import re
import justext
import logging
from cleantext import clean

goose = Goose()


def clean_text(text: str):
    return clean(text, fix_unicode=True, to_ascii=False, lower=False, no_emoji=True)

def formatting(text: str):
    text = re.sub(r"\n+", "\n", text).strip()
    list_sentences = text.split("\n")
    list_sentences = [clean_text(sentence) for sentence in list_sentences]
    if list_sentences == [""]:
        list_sentences = []
    return list_sentences


def deboilerplating(html: str):
    try:
        result_traf = trafilatura.extract(
            html,
            no_fallback=False,
            include_comments=False,
            include_tables=True,
            include_formatting=False,
        )
        if result_traf is not None:
            result_traf = formatting(result_traf)
        else:
            result_traf = []
    except Exception as e:
        result_traf = []
        logging.info(f"error calling trafilatura: {e}")

    try:
        result_goose = goose.extract(raw_html=html)
        result_goose = str(result_goose.cleaned_text)
        if result_goose is not None:
            result_goose = formatting(result_goose)
        else:
            result_goose = []
    except Exception as e:
        result_goose = []
        logging.info(f"error calling goose3: {e}")
    all_sentences = result_traf + result_goose
    sentence_set = list(dict.fromkeys(all_sentences))

    try:
        paragraphs = justext.justext(
            html, justext.get_stoplist("English"), 50, 200, 0.1, 0.2, 0.2, 200, True
        )
        invalid = [
            paragraph.text for paragraph in paragraphs if paragraph.is_boilerplate
        ]
    except Exception as e:
        invalid = []
        logging.info(f"error calling justext: {e}")

    sentence_set = [
        sentence
        for sentence in sentence_set
        if sentence not in invalid or sentence != ""
    ]

    if not sentence_set:
        sentence_set = []
        logging.info("Page not found")
    else:
        sentence_set = [
            sentence + " "
            if sentence.endswith((".", "!", "?", ".”"))
            else "\n" + sentence + "\n"
            for sentence in sentence_set
        ]

    return dict(
        {
            "set": sentence_set,
            "trafilatura": result_traf,
            "goose": result_goose,
            "justext": invalid,
        }
    )


def get_html(url: str) -> str:
    return trafilatura.fetch_url(url)

def get_text_from_html(html: str) -> str:
    result = deboilerplating(html)
    #print('result:', result)
    #print('')
    text = "".join(result["set"])
    pattern = r"\n+| \n"
    text = re.sub(pattern, "\n", text).strip()
    return text