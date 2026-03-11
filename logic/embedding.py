import multiprocessing
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from gensim.models import Word2Vec
from nltk.corpus import cess_esp
from nltk.corpus import brown
from datasets import load_dataset
from tqdm import tqdm
import operator
from logic.text_analysis import TextAnalysis
from logic.utils import Utils
from root import DIR_IMAGE, DIR_MODELS, DIR_EMBEDDING
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class Embedding(object):

    def __init__(self, lang='es'):
        self.lang = lang
        self.cores = multiprocessing.cpu_count()
        self.part_corpus = self.import_part_corpus(lang)
        self.text_analysis = TextAnalysis(lang)
        self.corpus = self.import_words_corpus()

    def import_words_corpus(self, max_samples=100000):
        """
        :Version: 1.1
        :Author: Edwin Puertas
        This function imports corpus in spanish, english, or french.
        Uses SemEval-2018 for es/en and Wikipedia via HuggingFace for other languages.
        :param max_samples: max Wikipedia articles to load (for languages without SemEval data)
        :type max_samples: int
        :rtype: list
        :return: list of text strings
        """
        result = []
        try:
            if self.lang in ('es', 'en'):
                file_es = 'SemEval-2018_AIT_DISC_ES.csv'
                file_en = 'SemEval-2018_AIT_DISC_EN.csv'
                file = file_es if self.lang == 'es' else file_en
                print('Loading.... {0} corpus'.format(file))
                corpus = self.text_analysis.import_corpus(file=file)
                result = [i[1] for i in corpus]
            else:
                print('Loading.... Wikipedia {0} corpus from HuggingFace'.format(self.lang))
                config_name = '20231101.{0}'.format(self.lang)
                dataset = load_dataset('wikimedia/wikipedia', config_name, split='train')
                articles = dataset['text'][:max_samples]
                for article in tqdm(articles):
                    import re
                    clean = re.sub(r'\s+', ' ', article).strip()
                    if len(clean) > 50:
                        result.append(clean)
                print('Loaded {0} texts from Wikipedia'.format(len(result)))
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error import_words_corpus: {0}'.format(e))
        return result

    def import_part_corpus(self, lang='es', max_samples=100000):
        """
        :Version: 1.1
        :Author: Edwin Puertas
        This function imports corpus in spanish, english, or french.
        Uses NLTK corpora for es/en and Wikipedia via HuggingFace for fr.
        :param lang: language code ('es', 'en', 'fr')
        :type lang: Text
        :param max_samples: max Wikipedia articles to load (for fr)
        :type max_samples: int
        :rtype: list
        :return: list of text strings
        """
        result = []
        try:
            if lang == 'es':
                print('Loading.... CESS corpus')
                sentences_list = cess_esp.sents()
                for sent in tqdm(list(sentences_list)):
                    list_text = [str(token).lower() for token in list(sent)]
                    text = ' '.join(list_text)
                    result.append(text)
            elif lang == 'en':
                print('Loading.... BROWN corpus')
                sentences_list = brown.sents(categories=['editorial'])
                for sent in tqdm(list(sentences_list)):
                    list_text = [str(token).lower() for token in list(sent)]
                    text = ' '.join(list_text)
                    result.append(text)
            else:
                print('Loading.... Wikipedia {0} corpus from HuggingFace'.format(lang))
                config_name = '20231101.{0}'.format(lang)
                dataset = load_dataset('wikimedia/wikipedia', config_name, split='train')
                articles = dataset['text'][:max_samples]
                for article in tqdm(articles):
                    import re
                    clean = re.sub(r'\s+', ' ', article).strip().lower()
                    sentences = re.split(r'[.!?\n]+', clean)
                    for sent in sentences:
                        sent = sent.strip()
                        if len(sent.split()) > 3:
                            result.append(sent)
                print('Loaded {0} sentences from Wikipedia'.format(len(result)))
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error import_part_corpus: {0}'.format(e))
        return result

    def words_embedding(self, model_name='word_embedding', size=300, min_count=50, window=5, sample=6e-5, negative=20,
                        alpha=0.03, min_alpha=0.0007):
        try:
            start_time = time.time()
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
            corpus_vec = self.text_analysis.part_vector(self.part_corpus, syllable=syllable)

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

    def get_similarity(self, model_name):
        dict_vocabulary = {}
        try:
            file_model = "{0}{1}_{2}.model".format(DIR_MODELS, model_name, self.lang)
            model = Word2Vec.load(file_model, mmap=None)
            vocabulary = list(model.wv.vocab)
            for i in vocabulary:
                dict_vocabulary[i] = model.most_similar(i)
                if i != '':
                    print('Token: {0}\nMost Similar:'.format(i))
                    for j in model.most_similar(i):
                        print(j)
            print(vocabulary)
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