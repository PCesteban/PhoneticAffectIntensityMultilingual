import os
import re
import sys
import logging
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
try:
    import pyphen
except Exception:
    pyphen = None
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
    _INVALID_PHONETIC_TOKENS = {' ', '', '\ufeff', '1'}

    def __init__(self, lang):
        lang_ipa = {'es': 'spa-Latn', 'en': 'eng-Latn', 'fr': 'fra-Latn'}
        lang_stemm = {'es': 'spanish', 'en': 'english', 'fr': 'french'}
        self.lang = lang
        # Epitran can emit very frequent warning logs (e.g., missing optional flite lookup in English),
        # which significantly slows long training runs due to console I/O.
        logging.getLogger('epitran').setLevel(logging.ERROR)
        logging.getLogger('epitran.lex_lookup').setLevel(logging.ERROR)
        self.stemmer = SnowballStemmer(language=lang_stemm[lang])
        self.epi = epitran.Epitran(lang_ipa[lang])
        self.en_pyphen = None
        if self.lang == 'en':
            if pyphen is not None:
                try:
                    self.en_pyphen = pyphen.Pyphen(lang='en_US')
                except Exception:
                    self.en_pyphen = None
            else:
                print('Warning: pyphen is not installed; EN syllable quality may degrade.')
        self.nlp = self.load_sapcy(lang)
        # Lightweight caches to reduce repeated transliteration/transcription cost.
        self._syllable_translit_cache = {}
        self._phoneme_list_cache = {}

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

    def _is_valid_phonetic_token(self, token):
        return token is not None and str(token).strip() not in self._INVALID_PHONETIC_TOKENS

    def _to_phonetic_token(self, text, cache_dict):
        if text is None:
            return ''
        text = str(text).strip()
        if not text:
            return ''

        if text in cache_dict:
            return cache_dict[text]

        phonetic = ''
        try:
            phonetic = self.epi.transliterate(text, normpunc=True)
        except Exception:
            phonetic = ''

        if not self._is_valid_phonetic_token(phonetic):
            try:
                list_phon = self.epi.trans_list(text, normpunc=True)
                list_phon = [i for i in list_phon if self._is_valid_phonetic_token(i)]
                if list_phon:
                    phonetic = ''.join(list_phon)
            except Exception:
                phonetic = ''

        if not self._is_valid_phonetic_token(phonetic):
            phonetic = ''

        cache_dict[text] = phonetic
        return phonetic

    def _clean_syllable_chunk(self, chunk):
        value = str(chunk).strip().lower()
        if self.lang == 'en':
            value = re.sub(r"[^a-z]", "", value)
        return value

    @staticmethod
    def _is_en_unsplit_syllables(syllables, token_text):
        token_text = str(token_text).strip().lower()
        if not token_text:
            return True
        if not syllables:
            return True
        if len(syllables) == 1:
            item = str(syllables[0]).strip().lower()
            if item == token_text and len(token_text) >= 4 and token_text.isalpha():
                return True
        return False

    def _get_en_pyphen_syllables(self, token_text):
        token_text = str(token_text).strip().lower()
        if not token_text or self.en_pyphen is None:
            return []
        try:
            inserted = self.en_pyphen.inserted(token_text)
        except Exception:
            return []
        if not inserted:
            return []
        chunks = [self._clean_syllable_chunk(i) for i in inserted.split('-')]
        return [i for i in chunks if i]

    def _get_token_syllables(self, token):
        token_text = str(token.text).strip().lower()
        source = 'none'
        unsplit_rejected = False

        try:
            raw_syllables = token._.syllables
        except Exception:
            raw_syllables = None

        if raw_syllables is None:
            syllables = []
        elif isinstance(raw_syllables, str):
            syllables = [raw_syllables]
        elif isinstance(raw_syllables, (list, tuple)):
            syllables = list(raw_syllables)
        else:
            syllables = []

        syllables = [self._clean_syllable_chunk(i) for i in syllables if str(i).strip()]
        syllables = [i for i in syllables if i]
        if syllables:
            source = 'spacy'

        if self.lang == 'en' and self._is_en_unsplit_syllables(syllables, token_text):
            pyphen_syllables = self._get_en_pyphen_syllables(token_text)
            if pyphen_syllables:
                syllables = pyphen_syllables
                source = 'pyphen'
            else:
                syllables = []
                source = 'none'
                unsplit_rejected = True

        return syllables, source, token_text, unsplit_rejected

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

    def part_vector(self, list_text, syllable=True, size_syllable=0, verbosity='summary', log_every=500):
        result = []
        try:
            processed_sentences = 0
            skipped_sentences = 0
            en_orthographic_syllable_fallbacks = 0
            en_spacy_syllable_tokens = 0
            en_pyphen_syllable_tokens = 0
            en_unsplit_rejected_tokens = 0

            # EN-only fast path for phoneme embeddings:
            # list_text is already sentence-level for legacy and wikipedia part corpora.
            # Skip spaCy parsing and convert each cleaned sentence directly to phoneme tokens.
            if (self.lang == 'en') and (syllable is False):
                for text in list_text:
                    try:
                        stm_text = self.clean_text(str(text).rstrip())
                        if not stm_text:
                            continue

                        processed_sentences += 1

                        if stm_text in self._phoneme_list_cache:
                            list_phonemes = self._phoneme_list_cache[stm_text]
                        else:
                            try:
                                list_phonemes = self.epi.trans_list(stm_text, normpunc=True)
                            except Exception:
                                list_phonemes = []
                            list_phonemes = [i for i in list_phonemes if self._is_valid_phonetic_token(i)]
                            self._phoneme_list_cache[stm_text] = list_phonemes

                        if list_phonemes:
                            result.append(list_phonemes)
                            if verbosity == 'full':
                                print('Sentence: {0}'.format(stm_text))
                                print('Vector: {0}'.format(list_phonemes))

                        if verbosity == 'summary' and processed_sentences % log_every == 0:
                            print('Processed {0} cleaned sentences; vectors generated: {1}'.format(
                                processed_sentences, len(result)
                            ))
                    except Exception:
                        skipped_sentences += 1
                        continue

                if verbosity == 'summary':
                    print('Completed part_vector. Processed {0} cleaned sentences; vectors generated: {1}'.format(
                        processed_sentences, len(result)
                    ))
                    if skipped_sentences > 0:
                        print('Skipped {0} sentences due to processing errors'.format(skipped_sentences))
                return result

            for text in list_text:
                doc = self.analysis_pipe(text.lower())
                if doc is None:
                    continue

                for stm in doc.sents:
                    try:
                        stm_text = self.clean_text(str(stm).rstrip())
                        if not stm_text:
                            continue

                        processed_sentences += 1

                        if syllable:
                            sentence_doc = self.analysis_pipe(stm_text)
                            if sentence_doc is None:
                                continue

                            list_syllable_phonetic = []
                            for token in sentence_doc:
                                token_syllables, syllable_source, token_text, unsplit_rejected = self._get_token_syllables(token)
                                if self.lang == 'en':
                                    if syllable_source == 'spacy':
                                        en_spacy_syllable_tokens += 1
                                    elif syllable_source == 'pyphen':
                                        en_pyphen_syllable_tokens += 1
                                    if unsplit_rejected:
                                        en_unsplit_rejected_tokens += 1

                                if not token_syllables:
                                    continue

                                token_emitted = False
                                n = len(token_syllables) if size_syllable == 0 else size_syllable
                                for s in token_syllables[:n]:
                                    if not s:
                                        continue
                                    syllable_phonetic = self._to_phonetic_token(s, self._syllable_translit_cache)
                                    if self._is_valid_phonetic_token(syllable_phonetic):
                                        list_syllable_phonetic.append(syllable_phonetic)
                                        token_emitted = True
                                    elif self.lang == 'en':
                                        # Keep EN output in syllable space even when phonetic conversion fails.
                                        # Fallback to orthographic syllable chunk; never fallback to full token.
                                        s_norm = str(s).strip().lower()
                                        if s_norm and s_norm.isalpha() and s_norm != token_text:
                                            list_syllable_phonetic.append(s_norm)
                                            token_emitted = True
                                            en_orthographic_syllable_fallbacks += 1

                            if list_syllable_phonetic:
                                result.append(list_syllable_phonetic)
                                if verbosity == 'full':
                                    print('Sentence: {0}'.format(stm_text))
                                    print('vector: {0}'.format(list_syllable_phonetic))
                        else:
                            if stm_text in self._phoneme_list_cache:
                                list_phonemes = self._phoneme_list_cache[stm_text]
                            else:
                                try:
                                    list_phonemes = self.epi.trans_list(stm_text, normpunc=True)
                                except Exception:
                                    list_phonemes = []
                                list_phonemes = [i for i in list_phonemes if self._is_valid_phonetic_token(i)]
                                self._phoneme_list_cache[stm_text] = list_phonemes

                            if list_phonemes:
                                result.append(list_phonemes)
                                if verbosity == 'full':
                                    print('Sentence: {0}'.format(stm_text))
                                    print('Vector: {0}'.format(list_phonemes))

                        if verbosity == 'summary' and processed_sentences % log_every == 0:
                            print('Processed {0} cleaned sentences; vectors generated: {1}'.format(
                                processed_sentences, len(result)
                            ))
                    except Exception:
                        skipped_sentences += 1
                        continue

            if verbosity == 'summary':
                print('Completed part_vector. Processed {0} cleaned sentences; vectors generated: {1}'.format(
                    processed_sentences, len(result)
                ))
                if skipped_sentences > 0:
                    print('Skipped {0} sentences due to processing errors'.format(skipped_sentences))
                if syllable and self.lang == 'en':
                    print('EN syllable sources -> spaCy tokens: {0}, pyphen tokens: {1}, unsplit rejected: {2}'.format(
                        en_spacy_syllable_tokens, en_pyphen_syllable_tokens, en_unsplit_rejected_tokens
                    ))
                    print('EN fallback usage -> orthographic syllable chunks: {0}'.format(
                        en_orthographic_syllable_fallbacks
                    ))
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
