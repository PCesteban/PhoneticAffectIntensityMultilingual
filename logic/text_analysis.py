import os
import re
import sys
import unicodedata
import spacy
from spacy.lang.es import Spanish
from spacy.lang.en import English
from spacy.lang.fr import French
from nltk import SnowballStemmer
try:
    from spacymoji import Emoji
except Exception:
    Emoji = None

try:
    from spacy_syllables import SpacySyllables
except Exception:
    SpacySyllables = None
import pandas as pd
import epitran
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from logic.steaming import Steaming
from logic.utils import Utils
from root import DIR_INPUT, DIR_EMBEDDING


class TextAnalysis(object):
    name = 'text_analysis'
    lang = 'es'

    def __init__(self, lang):
        lang_ipa = {'es': 'spa-Latn', 'en': 'eng-Latn', 'fr': 'fra-Latn'}
        lang_stemm = {'es': 'spanish', 'en': 'english', 'fr': 'french'}
        self.lang = lang
        self.stemmer = SnowballStemmer(language=lang_stemm[lang])
        self.epi = epitran.Epitran(lang_ipa[lang])
        self.nlp = self.load_sapcy(lang)

    def load_sapcy(self, lang):
        result = None
        try:
            spacy_models = {
                'es': 'es_core_news_md',
                'en': 'en_core_web_md',
                'fr': 'fr_core_news_md',
            }
            result = spacy.load(spacy_models[lang], disable=['ner'])
            stemmer_text = Steaming(lang)  # initialise component

            if SpacySyllables is not None:
                try:
                    syllables = SpacySyllables(result)
                    result.add_pipe(syllables, after="tagger")
                except Exception as e:
                    print('Warning: syllables component disabled: {0}'.format(e))

            if Emoji is not None:
                try:
                    emoji = Emoji(result)
                    result.add_pipe(emoji, first=True)
                except Exception as e:
                    print('Warning: emoji component disabled: {0}'.format(e))

            result.add_pipe(stemmer_text, after='parser', name='stemmer')
            print('Language: {0}\nText Analysis: {1}'.format(lang, result.pipe_names))
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error load_sapcy: {0}'.format(e))
        return result

    def analysis_pipe(self, text):
        doc = None
        try:
            doc = self.nlp(text.lower())
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error analysis_pipe: {0}'.format(e))
        return doc

    def sentences_vector(self, list_text):
        result = []
        try:
            setting = {'url': True, 'mention': True, 'emoji': False, 'hashtag': True, 'stopwords': True}
            for text in tqdm(list_text):
                text = self.clean_text(text, **setting)
                if text is not None:
                    doc = self.analysis_pipe(text)
                    if doc is not None:
                        vector = [i.text for i in doc]
                        result.append(vector)
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error sentences_vector: {0}'.format(e))
        return result

    def part_vector(self, list_text, syllable=True, size_syllable=0):
        result = []
        try:
            for text in list_text:
                doc = self.analysis_pipe(text.lower())
                for stm in doc.sents:
                    stm = str(stm).rstrip()
                    stm = self.clean_text(stm)
                    if stm != '':
                        print('Sentence: {0}'.format(stm))
                        if syllable:
                            list_syllable = [token['syllables'] for token in self.tagger(stm) if
                                             token['syllables'] is not None]
                            list_syllable_phonetic = []
                            for token_syllables in list_syllable:
                                n = len(token_syllables) if size_syllable == 0 else size_syllable
                                for s in token_syllables[:n]:
                                    syllable_phonetic = self.epi.transliterate(s, normpunc=True)
                                    if syllable_phonetic not in [' ', '', '\ufeff', '1']:
                                        list_syllable_phonetic.append(syllable_phonetic)
                            result.append(list_syllable_phonetic)
                            print('vector: {0}'.format(list_syllable_phonetic))
                        else:
                            list_phonemes = self.epi.trans_list(stm, normpunc=True)
                            list_phonemes = [i for i in list_phonemes if i not in [' ', '', '\ufeff', '1']]
                            result.append(list_phonemes)
                            print('Vector: {0}'.format(list_phonemes))
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error phonemes_vector: {0}'.format(e))
        return result

    def tagger(self, text):
        result = None
        try:
            list_tagger = []
            doc = self.analysis_pipe(text.lower())
            for token in doc:
                item = {'text': token.text, 'lemma': token.lemma_, 'stem': token._.stem, 'pos': token.pos_,
                        'tag': token.tag_, 'dep': token.dep_, 'shape': token.shape_, 'is_alpha': token.is_alpha,
                        'is_stop': token.is_stop, 'is_digit': token.is_digit, 'is_punct': token.is_punct,
                        'syllables': token._.syllables}
                list_tagger.append(item)
            result = list_tagger
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error tagger: {0}'.format(e))
        return result

    def dependency(self, text):
        result = []
        try:
            doc = self.analysis_pipe(text.lower())
            doc_chunks = list(doc.noun_chunks)
            for chunk in doc_chunks:
                item = {'chunk': chunk, 'text': chunk.text,
                        'root_text': chunk.root.text, 'root_dep': chunk.root.dep_}
                result.append(item)
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error dependency: {0}'.format(e))
        return result

    def dependency_all(self, text):
        result = []
        try:
            doc = self.analysis_pipe(text.lower())
            for chunk in doc.noun_chunks:
                item = {'chunk': chunk, 'text': chunk.root.text, 'pos_': chunk.root.pos_, 'dep_': chunk.root.dep_,
                        'tag_': chunk.root.tag_, 'lemma_': chunk.root.lemma_, 'is_stop': chunk.root.is_stop,
                        'is_punct': chunk.root.is_punct, 'head_text': chunk.root.head.text,
                        'head_pos': chunk.root.head.pos_,
                        'children': [{'child': child, 'pos_': child.pos_, 'dep_': child.dep_,
                                      'tag_': child.tag_, 'lemma_': child.lemma_, 'is_stop': child.is_stop,
                                      'is_punct': child.is_punct, 'head.text': child.head.text,
                                      'head.pos_': child.head.pos_} for child in chunk.root.children]}
                result.append(item)
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error dependency_all: {0}'.format(e))
        return result

    def dependency_child(self, text):
        result = []
        try:
            doc = self.analysis_pipe(text.lower())
            for token in doc:
                item = {'chunk': token.text, 'text': token.text, 'pos_': token.pos_,
                        'dep_': token.dep_, 'tag_': token.tag_, 'head_text': token.head.text,
                        'head_pos': token.head.pos_, 'children': None}
                if len(list(token.children)) > 0:
                    item['children'] = [{'child': child, 'pos_': child.pos_, 'dep_': child.dep_,
                                         'tag_': child.tag_, 'head.text': child.head.text,
                                         'head.pos_': child.head.pos_} for child in token.children]
                result.append(item)
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error dependency_child: {0}'.format(e))
        return result

    def dependency_tree(self, text):
        result = []
        try:
            doc = self.analysis_pipe(text.lower())
            root = [token for token in doc if token.head == token][0]
            if len(list(root.lefts)) > 0:
                subject = list(root.lefts)[0]
                for descendant in subject.subtree:
                    assert subject is descendant or subject.is_ancestor(descendant)
                    item = {}
                    item['text'] = descendant.text
                    item['dep'] = descendant.dep_
                    item['n_lefts'] = descendant.n_lefts
                    item['n_rights'] = descendant.n_rights
                    item['descendant'] = [ancestor.text for ancestor in descendant.ancestors]
                    result.append(item)
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error dependency_tree: {0}'.format(e))
        return result

    def import_corpus(self, file, sep=';', name_id="id", name_text="text"):
        result = []
        try:
            count = 0
            file = DIR_INPUT + file
            df = pd.read_csv(file, sep=sep)
            df.dropna(inplace=True)
            df = df[[name_id, name_text]].values.tolist()
            for row in tqdm(df):
                id = row[0]
                text = str(row[1])
                if len(text) > 0 or text != '':
                    result.append([id, text])
                    count = count + 1
            print('# Sentence: {0}'.format(count))
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error import_corpus: {0}'.format(e))
        return result

    @staticmethod
    def import_dataset(file, **kwargs):
        result = None
        try:
            print('Loading dataset {0}...'.format(file))
            setting = {}
            mini_size = kwargs.get('mini_size') if type(kwargs.get('mini_size')) is int else 2
            sep = ';' if type(kwargs.get('sep')) is str else kwargs.get('sep')
            setting['url'] = kwargs.get('url') if type(kwargs.get('url')) is bool else False
            setting['mention'] = kwargs.get('mention') if type(kwargs.get('mention')) is bool else False
            setting['emoji'] = kwargs.get('emoji') if type(kwargs.get('emoji')) is bool else False
            setting['hashtag'] = kwargs.get('hashtag') if type(kwargs.get('hashtag')) is bool else False
            setting['lemmatize'] = kwargs.get('lemmatizer') if type(kwargs.get('lemmatizer')) is bool else False
            setting['stopwords'] = kwargs.get('stopwords') if type(kwargs.get('stopwords')) is bool else False
            data = []
            file_path = DIR_INPUT + file
            raw_data = pd.read_csv(file_path, sep=sep, encoding='UTF-8')
            for i, row in raw_data.iterrows():
                text = TextAnalysis.clean_text(row['Tweet'], **setting)
                len_text = len(text.split(' '))
                if len_text > mini_size:
                    tag = int(row['Intensity'])
                    value = 0
                    if tag > 0:
                        value = 1
                    elif tag < 0:
                        value = -1
                    elif tag == 0:
                        value = 0
                    data.append([text, value])
            result = pd.DataFrame(data, columns=['message', 'valence'])
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error import_dataset: {0}'.format(e))
        return result

    @staticmethod
    def proper_encoding(text):
        result = ''
        try:
            text = unicodedata.normalize('NFD', text)
            text = text.encode('ascii', 'ignore')
            result = text.decode("utf-8")
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error proper_encoding: {0}'.format(e))
        return result

    @staticmethod
    def stopwords(text):
        result = ''
        try:
            lang_nlp = {'es': Spanish, 'en': English, 'fr': French}
            nlp = lang_nlp.get(TextAnalysis.lang, English)()
            doc = nlp(text)
            token_list = [token.text for token in doc]
            sentence = []
            for word in token_list:
                lexeme = nlp.vocab[word]
                if not lexeme.is_stop:
                    sentence.append(word)
            result = ' '.join(sentence)
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error stopwords: {0}'.format(e))
        return result

    def lemmatization(self, text):
        result = ''
        list_tmp = []
        try:
            doc = TextAnalysis.analysis_pipe(text.lower())
            for token in doc:
                list_tmp.append(str(token.lemma_))
            result = ' '.join(list_tmp)
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error lemmatization: {0}'.format(e))
        return result

    def stemming(self, text):
        try:
            tokens = word_tokenize(text)
            stemmed = [self.stemmer.stem(word) for word in tokens]
            text = ' '.join(stemmed)
            return text
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error stemming: {0}'.format(e))
            return None

    @staticmethod
    def delete_special_patterns(text):
        result = ''
        try:
            text = re.sub(r'\©|\×|\⇔|\_|\»|\«|\~|\#|\$|\€|\Â|\�|\¬', ' ', text)# Elimina caracteres especilaes
            text = re.sub(r'\,|\;|\:|\!|\¡|\’|\‘|\”|\“|\"|\'|\`', ' ', text)# Elimina puntuaciones
            text = re.sub(r'\}|\{|\[|\]|\(|\)|\<|\>|\?|\¿|\°|\|', ' ', text)  # Elimina parentesis
            text = re.sub(r'\/|\-|\+|\*|\=|\^|\%|\&|\$|\.', ' ', text)  # Elimina operadores
            text = re.sub(r'\b\d+(?:\.\d+)?\s+', ' ', text)  # Elimina número con puntuacion
            result = text.lower()
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error delete_special_patterns: {0}'.format(e))
        return result

    @staticmethod
    def clean_text(text, **kwargs):
        result = ''
        try:
            url = kwargs.get('url') if type(kwargs.get('url')) is bool else False
            mention = kwargs.get('mention') if type(kwargs.get('mention')) is bool else False
            emoji = kwargs.get('emoji') if type(kwargs.get('emoji')) is bool else False
            hashtag = kwargs.get('hashtag') if type(kwargs.get('hashtag')) is bool else False
            lemmatizer = kwargs.get('lemmatizer') if type(kwargs.get('lemmatizer')) is bool else False
            stopwords = kwargs.get('stopwords') if type(kwargs.get('stopwords')) is bool else False

            text_out = str(text).lower()
            text_out = re.sub("[\U0001f000-\U000e007f]", ' ', text_out) if emoji else text_out
            text_out = re.sub(
                r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+'
                r'|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
                ' ', text_out) if url else text_out
            text_out = re.sub("@([A-Za-z0-9_]{1,40})", ' ', text_out) if mention else text_out
            text_out = re.sub("#([A-Za-z0-9_]{1,40})", ' ', text_out) if hashtag else text_out
            text_out = TextAnalysis.delete_special_patterns(text_out)
            text_out = TextAnalysis.lemmatization(text_out) if lemmatizer else text_out
            text_out = TextAnalysis.stopwords(text_out) if stopwords else text_out
            text_out = re.sub(r'\s+', ' ', text_out).strip()
            text_out = text_out.rstrip()
            result = text_out if text_out != ' ' else None
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error clean_text: {0}'.format(e))
        return result

    @staticmethod
    def import_lexicon_vad(path_input, lang='es'):
        try:
            result = {}
            with open(path_input, encoding='utf-8') as intput:
                line_list = intput.readlines()
                for line in line_list[1:]:
                    item = line.strip('\n').split('\t')
                    word = str(item[1]) if lang == 'es' else str(item[0])
                    if word != 'NO TRANSLATION' and word:
                        valence = float(item[2])
                        arousal = float(item[3])
                        dominance = float(item[4])
                        vad = valence + arousal + dominance
                        if word in result:
                            value = list(result[word])
                            valence = (valence + value[0]) / 2
                            arousal = (arousal + value[1]) / 2
                            dominance = (dominance + value[2]) / 2
                            result[word] = [round(valence, 4), round(arousal, 4), round(dominance, 4), round(vad, 4)]
                        else:
                            result[word] = [round(valence, 4), round(arousal, 4), round(dominance, 4), round(vad, 4)]
            return result
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error import_lexicon_vad: {0}'.format(e))
        finally:
            intput.close()

    @staticmethod
    def token_frequency(model_name, corpus_vec):
        dict_token = {}
        try:
            sep = os.sep
            file_output = DIR_EMBEDDING + 'frequency' + sep + 'frequency_' + model_name + '.csv'
            for list_tokens in corpus_vec:
                for token in list_tokens:
                    if token not in [' ', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']:
                        if token in dict_token:
                            value = dict_token[token]
                            dict_token[token] = value + 1
                        else:
                            dict_token[token] = 1
            list_token = [{'token': k, 'freq': v} for k, v in dict_token.items()]
            df = pd.DataFrame(list_token, columns=['token', 'freq'])
            df.to_csv(file_output, encoding="utf-8", sep=";", index=False)
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error token_frequency: {0}'.format(e))
        return dict_token
