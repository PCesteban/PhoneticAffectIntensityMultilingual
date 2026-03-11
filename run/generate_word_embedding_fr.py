
from logic.embedding import Embedding

em = Embedding(lang='fr')
em.words_embedding(model_name='word_embedding')
em.plot('word_embedding_fr')
em.plot_clusters('word_embedding_fr')
em.get_similarity(model_name='word_embedding')
