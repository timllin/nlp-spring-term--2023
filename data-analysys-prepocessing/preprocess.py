import re
import emoji
from unidecode import unidecode
import html as ihtml
from bs4 import BeautifulSoup

def remove_repeated_puncs(x):
    x_ = re.sub(r'\s([?.!:,;](?:\s|$))', r'\1', x)
    x_ = re.sub(r'(\W)\s?(?=\1)', '', x_)
    x_ = re.sub("\[(.[0-9]{1,2})\]", '', x_)
    return x_

def clean_text(x):
    text = unidecode(x)
    if ("<" in text) or (">" in text):
        text = BeautifulSoup(ihtml.unescape(text), features="lxml").text
    text = emoji.replace_emoji(text)
    text = text.lower()
    text = remove_repeated_puncs(text)
    if len(text.split()) <= 3:
        text = " ".join(text.split("_"))
    text = " ".join(text.split())
    text = text.strip("#$%&\*+-/:;=@\\^_`|~ ").rstrip('., ').lstrip("!? ")
    if (len(text) > 2) & (text.count('"') == 2):
        if (text[0] == '"') & (text[-1] == '"'):
            text = text.strip('" ')
    return text

