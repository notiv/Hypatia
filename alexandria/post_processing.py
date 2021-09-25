import json
import requests
import urllib
import warnings
from cleantext import clean


def get_api_key(filename='api_key.txt'):
    with open(filename) as f:
        return f.readline()


def clean_up_text(input_text, discard_small_words_thresh=3):
    input_text = " ".join(input_text.split())
    input_text = " ".join(word for word in input_text.split() if len(word)>discard_small_words_thresh)
    input_text = clean(input_text,
        fix_unicode=True,               # fix various unicode errors
        to_ascii=True,                  # transliterate to closest ASCII representation
        lower=True,                     # lowercase text
        no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
        no_punct=True
    )
    return input_text


def search_book(input_text, api_key, n_results=5):
    clean_text_enc = urllib.parse.quote(input_text)
    query = f'https://www.googleapis.com/books/v1/volumes?q={clean_text_enc}&key={api_key}'
    response = requests.get(query)
    if response.status_code != 200:
        return None
    json = response.json()
    n_items = json.get("totalItems")
    if n_items == 0:
        warnings.warn(f"No items returned from the API.")
        return None
    titles = []
    for i in range(min(n_items, n_results)):
        item = json.get("items")[i]
        volume_info = item.get('volumeInfo')
        title = volume_info.get('title')
        titles.append(title)
    return titles


if __name__ == '__main__':
    # user input
    input_text = 'fastai deep learning'
    api_key = get_api_key('api_key.txt')

    titles = search_book(input_text, api_key)

    print(titles)

