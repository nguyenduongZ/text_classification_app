import re
import nltk

from autocorrect import Speller
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

spell = Speller(lang='en')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_html(text):
    rules = [
        {r'>\s+': u'>'},
        {r'\s+': u' '},
        {r'\s*<br\s*/?>\s*': u'\n'},
        {r'</(div)\s*>\s*': u'\n'},
        {r'</(p|h\d)\s*>\s*': u'\n\n'},
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},
        {r'[ \t]*<[^<]*?/?>': u''},
        {r'^\s+': u''}
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
    return text

def preprocess_text(text, use_spell=True, use_stem=False, use_lemma=True):
    text = clean_html(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    tokens = word_tokenize(text, preserve_line=True)
    tokens = [t for t in tokens if t not in stop_words]

    if use_spell:
        tokens = [spell(t) for t in tokens]

    if use_stem:
        tokens = [stemmer.stem(t) for t in tokens]

    if use_lemma:
        tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]

    return ' '.join(tokens)

def preprocess_corpus(texts, **kwargs):
    return [preprocess_text(text, **kwargs) for text in texts]

def preprocess_text_for_lstm(text, vocab, use_spell=True, use_stem=False, use_lemma=True):
    text = clean_html(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    tokens = word_tokenize(text, preserve_line=True)
    tokens = [t for t in tokens if t not in stop_words]

    if use_spell:
        tokens = [spell(t) for t in tokens]

    if use_stem:
        tokens = [stemmer.stem(t) for t in tokens]

    if use_lemma:
        tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]

    ids = [vocab.get(token, vocab.get('<UNK>', 0)) for token in tokens]

    return ids

def preprocess_text_for_transformer(text):
    text = clean_html(text)
    text = text.lower()
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_corpus_for_transformer(texts):
    return [preprocess_text_for_transformer(text) for text in texts]