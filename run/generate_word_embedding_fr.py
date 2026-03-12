
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.embedding import Embedding

em = Embedding(lang='fr', max_samples=20000)
em.words_embedding(model_name='word_embedding', size=150)
em.plot('word_embedding_fr')
em.plot_clusters('word_embedding_fr')
em.get_similarity(model_name='word_embedding')
