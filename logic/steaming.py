from spacy.tokens import Doc, Span, Token
from nltk.stem.snowball import SnowballStemmer


class Steaming(object):
    name = 'stemmer'

    def __init__(self, lang):
        dict_lang = {'es': 'spanish', 'en': 'english', 'fr': 'french'}
        self.stemmer = SnowballStemmer(dict_lang[lang])
        Token.set_extension('stem', default='', force=True)

    def __call__(self, doc):
        for token in doc:
            if not token.is_punct and not token.is_stop and not token.is_digit:
                token._.set('stem', self.stemmer.stem(token.text))
        return doc