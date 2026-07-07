import re
import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


def remove_stopwords(text: str) -> str:
    return ' '.join(word for word in text.split() if word.lower() not in stop_words)


def remove_punctuation(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text)


def remove_special_chars(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)


def remove_extra_spaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def clean_text(text: str) -> str:
    """Full cleaning pipeline: lowercase -> remove stopwords -> remove punctuation
    -> remove special chars -> collapse extra whitespace."""
    text = text.lower()
    text = remove_stopwords(text)
    text = remove_punctuation(text)
    text = remove_special_chars(text)
    text = remove_extra_spaces(text)
    return text