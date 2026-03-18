import multiprocessing
import json
import os
import re
import sys
import time
import traceback
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import spacy
from gensim.models import Word2Vec
from nltk.corpus import cess_esp
from nltk.corpus import brown
from datasets import load_dataset
from tqdm import tqdm
import operator
from logic.text_analysis import TextAnalysis
from logic.utils import Utils
from root import DIR_IMAGE, DIR_MODELS, DIR_EMBEDDING, DIR_OUTPUT
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# datasets uses spacy.Language in its dill helpers; spaCy 2 exposes it under spacy.language.Language.
if not hasattr(spacy, "Language"):
    from spacy.language import Language as _SpacyLanguage

    spacy.Language = _SpacyLanguage


class Embedding(object):

    def __init__(
        self,
        lang='es',
        max_samples=100000,
        word_max_samples=None,
        part_max_samples=None,
        verbosity='full',
        corpus_source='legacy',
    ):
        self.lang = lang
        self.max_samples = max_samples
        self.word_max_samples = max_samples if word_max_samples is None else word_max_samples
        self.part_max_samples = max_samples if part_max_samples is None else part_max_samples
        self.verbosity = verbosity
        self.corpus_source = corpus_source
        self.cores = multiprocessing.cpu_count()
        self.text_analysis = TextAnalysis(lang)
        # Lazy-load corpora only when needed by the selected training method.
        self.part_corpus = None
        self.corpus = None

    def _ensure_word_corpus_loaded(self):
        if self.corpus is None:
            self.corpus = self.import_words_corpus(
                max_samples=self.word_max_samples,
                corpus_source=self.corpus_source,
            )

    def _ensure_part_corpus_loaded(self):
        if self.part_corpus is None:
            self.part_corpus = self.import_part_corpus(
                lang=self.lang,
                max_samples=self.part_max_samples,
                corpus_source=self.corpus_source,
            )

    @staticmethod
    def _wikipedia_config_name(lang):
        return '20231101.{0}'.format(lang)

    @staticmethod
    def _wikipedia_cache_file(lang, max_samples):
        cache_dir = os.path.join(DIR_EMBEDDING, 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        file_name = 'wikipedia_{0}_{1}_{2}.jsonl'.format('20231101', lang, max_samples)
        return os.path.join(cache_dir, file_name)

    @staticmethod
    def _load_cached_wikipedia_articles(cache_file):
        result = []
        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                result.append(row.get('text', ''))
        return result

    @staticmethod
    def _save_cached_wikipedia_articles(cache_file, articles):
        with open(cache_file, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps({'text': article}, ensure_ascii=False) + '\n')

    def _stream_wikipedia_articles(self, lang, max_samples):
        if max_samples is None:
            raise ValueError('max_samples must be provided for Wikipedia streaming mode.')

        config_name = self._wikipedia_config_name(lang)
        dataset = load_dataset(
            'wikimedia/wikipedia',
            config_name,
            split='train',
            streaming=True,
        )
        result = []
        progress = tqdm(total=max_samples, desc='Streaming Wikipedia subset ({0})'.format(lang))
        for row in dataset:
            result.append(row.get('text', ''))
            progress.update(1)
            if len(result) >= max_samples:
                break
        progress.close()
        return result

    def _get_wikipedia_articles(self, lang, max_samples=100000):
        cache_file = self._wikipedia_cache_file(lang=lang, max_samples=max_samples)
        if os.path.exists(cache_file):
            print('Using cached Wikipedia subset: {0}'.format(cache_file))
            articles = self._load_cached_wikipedia_articles(cache_file)
            print('Loaded {0} cached Wikipedia articles'.format(len(articles)))
            return articles

        print('Streaming.... Wikipedia {0} corpus from HuggingFace'.format(lang))
        print('Wikipedia sample cap: {0}'.format(max_samples))
        articles = self._stream_wikipedia_articles(lang=lang, max_samples=max_samples)
        self._save_cached_wikipedia_articles(cache_file=cache_file, articles=articles)
        print('Saved cached Wikipedia subset: {0}'.format(cache_file))
        print('Loaded {0} streamed Wikipedia articles'.format(len(articles)))
        return articles

    def _import_words_from_wikipedia(self, lang, max_samples=100000):
        result = []
        articles = self._get_wikipedia_articles(lang=lang, max_samples=max_samples)
        for article in tqdm(articles, desc='Preparing word corpus ({0})'.format(lang)):
            clean = re.sub(r'\s+', ' ', article).strip()
            if len(clean) > 50:
                result.append(clean)
        print('Loaded {0} texts from Wikipedia'.format(len(result)))
        return result

    def _import_part_from_wikipedia(self, lang, max_samples=100000):
        result = []
        articles = self._get_wikipedia_articles(lang=lang, max_samples=max_samples)
        for article in tqdm(articles, desc='Preparing part corpus ({0})'.format(lang)):
            clean = re.sub(r'\s+', ' ', article).strip().lower()
            sentences = re.split(r'[.!?\n]+', clean)
            for sent in sentences:
                sent = sent.strip()
                if len(sent.split()) > 3:
                    result.append(sent)
        print('Loaded {0} sentences from Wikipedia'.format(len(result)))
        return result

    def import_words_corpus(self, max_samples=100000, corpus_source=None):
        """
        :Version: 1.1
        :Author: Edwin Puertas
        This function imports corpus in spanish, english, or french.
        Source is selected with corpus_source:
          - legacy: SemEval-2018 for es/en, Wikipedia fallback for other languages
          - wikipedia: Wikipedia via HuggingFace for all languages
        :param max_samples: max Wikipedia articles to load (for languages without SemEval data)
        :type max_samples: int
        :param corpus_source: data source strategy ('legacy' or 'wikipedia')
        :type corpus_source: Text
        :rtype: list
        :return: list of text strings
        """
        result = []
        try:
            source = self.corpus_source if corpus_source is None else corpus_source
            if source not in ('legacy', 'wikipedia'):
                raise ValueError("Invalid corpus_source '{0}'. Use 'legacy' or 'wikipedia'.".format(source))

            if source == 'legacy' and self.lang in ('es', 'en'):
                file_es = 'SemEval-2018_AIT_DISC_ES.csv'
                file_en = 'SemEval-2018_AIT_DISC_EN.csv'
                file = file_es if self.lang == 'es' else file_en
                print('Loading.... {0} corpus'.format(file))
                corpus = self.text_analysis.import_corpus(file=file)
                result = [i[1] for i in corpus]
            else:
                result = self._import_words_from_wikipedia(lang=self.lang, max_samples=max_samples)
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error import_words_corpus: {0}'.format(e))
            traceback.print_exc()
        return result

    def import_part_corpus(self, lang='es', max_samples=100000, corpus_source=None):
        """
        :Version: 1.1
        :Author: Edwin Puertas
        This function imports corpus in spanish, english, or french.
        Source is selected with corpus_source:
          - legacy: NLTK CESS(es)/BROWN(en), Wikipedia fallback for fr
          - wikipedia: Wikipedia via HuggingFace for all languages
        :param lang: language code ('es', 'en', 'fr')
        :type lang: Text
        :param max_samples: max Wikipedia articles to load (for fr)
        :type max_samples: int
        :param corpus_source: data source strategy ('legacy' or 'wikipedia')
        :type corpus_source: Text
        :rtype: list
        :return: list of text strings
        """
        result = []
        try:
            source = self.corpus_source if corpus_source is None else corpus_source
            if source not in ('legacy', 'wikipedia'):
                raise ValueError("Invalid corpus_source '{0}'. Use 'legacy' or 'wikipedia'.".format(source))

            if source == 'legacy' and lang == 'es':
                print('Loading.... CESS corpus')
                sentences_list = cess_esp.sents()
                for sent in tqdm(list(sentences_list)):
                    list_text = [str(token).lower() for token in list(sent)]
                    text = ' '.join(list_text)
                    result.append(text)
            elif source == 'legacy' and lang == 'en':
                print('Loading.... BROWN corpus')
                sentences_list = brown.sents(categories=['editorial'])
                for sent in tqdm(list(sentences_list)):
                    list_text = [str(token).lower() for token in list(sent)]
                    text = ' '.join(list_text)
                    result.append(text)
            else:
                result = self._import_part_from_wikipedia(lang=lang, max_samples=max_samples)
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error import_part_corpus: {0}'.format(e))
            traceback.print_exc()
        return result

    def words_embedding(self, model_name='word_embedding', size=300, min_count=50, window=5, sample=6e-5, negative=20,
                        alpha=0.03, min_alpha=0.0007):
        try:
            if self.lang == 'fr' and size != 150:
                print('Forcing French word embedding size to 150 for PETER compatibility')
                size = 150

            start_time = time.time()
            self._ensure_word_corpus_loaded()
            corpus_vec = self.text_analysis.sentences_vector(self.corpus)

            model = Word2Vec(corpus_vec, cbow_mean=1, workers=self.cores-1, size=size, min_count=min_count,
                             window=window, sample=sample, negative=negative, alpha=alpha, min_alpha=min_alpha, iter=10)

            model_name = model_name + '_' + self.lang
            file_name = DIR_MODELS + model_name + '.model'
            model.save(file_name)
            print('Model {0} generated successful!'.format(model_name))

            vocabulary = list(model.wv.vocab)
            print('Vocabulary: {0}'.format(vocabulary))

            self.text_analysis.token_frequency(model_name=model_name, corpus_vec=corpus_vec)
            # Calculated Time processing
            t_sec = round(time.time() - start_time)
            (t_min, t_sec) = divmod(t_sec, 60)
            (t_hour, t_min) = divmod(t_min, 60)
            time_processing = '{} hour:{} min:{} sec'.format(t_hour, t_min, t_sec)
            print('Time Processing: {}'.format(time_processing))
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error words_embedding: {0}'.format(e))

    def part_embedding(self, model_name='part_embedding', size=150, min_count=10, window=5, sample=6e-5, negative=20,
                       alpha=0.03, min_alpha=0.0007, syllable=True):
        """
        :Version: 1.0
        :Author: Edwin Puertas
        This function generated phonemes embedding in spanish and english.
        :param list_doc: list of documents (corpus)
        :type list: Text
        :rtype: dict
        :return: terms by documents
        """
        try:
            start_time = time.time()
            self._ensure_part_corpus_loaded()
            corpus_vec = self.text_analysis.part_vector(
                self.part_corpus,
                syllable=syllable,
                verbosity=self.verbosity,
            )

            model = Word2Vec(corpus_vec, cbow_mean=1, workers=self.cores-1, size=size, min_count=min_count,
                             window=window, sample=sample, negative=negative, alpha=alpha, min_alpha=min_alpha, iter=10)

            model_name = model_name + '_' + self.lang
            file_name = DIR_MODELS + model_name + '.model'
            model.save(file_name)
            print('Model {0} generated successful!'.format(file_name))

            vocabulary = list(model.wv.vocab)
            print('Vocabulary: {0}'.format(vocabulary))

            self.text_analysis.token_frequency(model_name=model_name, corpus_vec=corpus_vec)
            #Calculated Time processing
            t_sec = round(time.time() - start_time)
            (t_min, t_sec) = divmod(t_sec, 60)
            (t_hour, t_min) = divmod(t_min, 60)
            time_processing = '{} hour:{} min:{} sec'.format(t_hour, t_min, t_sec)
            print('Time Processing: {}'.format(time_processing))
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error part_embedding: {0}'.format(e))

    def get_similarity(self, model_name, topn=10, verbosity='full', save_output=False):
        dict_vocabulary = {}
        try:
            file_model = "{0}{1}_{2}.model".format(DIR_MODELS, model_name, self.lang)
            model = Word2Vec.load(file_model, mmap=None)
            vocabulary = list(model.wv.vocab)
            records = []

            for i in vocabulary:
                similar_items = model.most_similar(i, topn=topn)
                dict_vocabulary[i] = similar_items
                records.append({'token': i, 'most_similar': similar_items})

                if verbosity == 'full' and i != '':
                    print('Token: {0}\nMost Similar:'.format(i))
                    for j in similar_items:
                        print(j)

            if verbosity == 'summary':
                print('Similarity computed for {0} tokens (topn={1})'.format(len(vocabulary), topn))
                for token in vocabulary[:10]:
                    print('{0}: {1}'.format(token, dict_vocabulary[token][:3]))

            if save_output:
                os.makedirs(DIR_OUTPUT, exist_ok=True)
                base = '{0}_{1}'.format(model_name, self.lang)
                json_path = os.path.join(DIR_OUTPUT, 'similarity_{0}.json'.format(base))
                csv_path = os.path.join(DIR_OUTPUT, 'similarity_{0}.csv'.format(base))

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)

                csv_rows = []
                for row in records:
                    token = row['token']
                    for similar_token, score in row['most_similar']:
                        csv_rows.append({
                            'token': token,
                            'similar_token': similar_token,
                            'score': score,
                        })
                pd.DataFrame(csv_rows).to_csv(csv_path, index=False, sep=';', encoding='utf-8')
                print('Saved similarity outputs: {0}, {1}'.format(json_path, csv_path))
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_similarity: {0}'.format(e))
        return dict_vocabulary

    @staticmethod
    def plot(model_name, size=15):
        try:
            print('Plot {0} embedding...'.format(model_name))
            sep = os.sep
            #Creates and TSNE model and plots it
            file_model = DIR_MODELS + model_name + ".model"
            model = Word2Vec.load(file_model, mmap=None)
            labels = []
            tokens = []
            list_vocabulary = list(model.wv.vocab)
            for word in list_vocabulary:
                tokens.append(model[word])
                labels.append(word)

            tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
            new_values = tsne_model.fit_transform(tokens)

            x = []
            y = []
            for value in new_values:
                x.append(value[0])
                y.append(value[1])

            plt.figure(figsize=(size, size))
            for i in range(len(x)):
                plt.scatter(x[i], y[i], marker='X', color='blue')
                plt.annotate(labels[i],
                             xy=(x[i], y[i]),
                             xytext=(5, 2),
                             textcoords='offset points',
                             ha='right',
                             va='bottom')
            file_output = DIR_EMBEDDING + sep + 'images' + sep + model_name + '.png'
            plt.savefig(file_output)
            plt.show()
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error plot: {0}'.format(e))

    def plot_clusters(self, model_name):
        try:
            sep = os.sep
            file_model = DIR_MODELS + model_name + ".model"
            model = Word2Vec.load(file_model, mmap=None)
            vocab = list(model.wv.vocab)
            file_words = DIR_EMBEDDING + 'frequency' + sep + 'frequency_' + model_name + '.csv'
            df = pd.read_csv(file_words,  delimiter=';')
            dict_token = dict(df.values.tolist())
            keys = dict(sorted(dict_token.items(), key=operator.itemgetter(1)))
            if len(keys) > 10:
                cluster_keys = []
                count = 1
                for k, v in keys.items():
                    if count == 10:
                        break
                    else:
                        if k in vocab:
                            cluster_keys.append(k)
                            count += 1
                embedding_clusters = []
                word_clusters = []
                for word in cluster_keys:
                    embeddings = []
                    words = []
                    for similar_word, _ in model.most_similar(word, topn=30):
                        words.append(similar_word)
                        embeddings.append(model[similar_word])
                    embedding_clusters.append(embeddings)
                    word_clusters.append(words)

                embedding_clusters = np.array(embedding_clusters)
                n, m, k = embedding_clusters.shape
                tsne_model = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
                embeddings_model = np.array(tsne_model.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
                plt.figure(figsize=(16, 9))
                colors = cm.rainbow(np.linspace(0, 1, len(cluster_keys)))

                for label, embeddings, words, color in zip(cluster_keys, embeddings_model, word_clusters, colors):
                    x = embeddings[:, 0]
                    y = embeddings[:, 1]

                    plt.scatter(x, y, c=color, alpha=0.5, label=label)
                    for i, word in enumerate(words):
                        plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                                     textcoords='offset points', ha='right',
                                     va='bottom', size=8)
                plt.legend(loc=4)
                plt.title('Cluster Embedding - ' + model_name)
                plt.grid(True)
                file_output = DIR_EMBEDDING + sep + 'images' + sep + 'cluster_' + model_name + '.png'
                plt.savefig(file_output, format='png', dpi=150, bbox_inches='tight')
                plt.show()

        except Exception as e:
            print('Error plot_clusters: {0}'.format(e))
