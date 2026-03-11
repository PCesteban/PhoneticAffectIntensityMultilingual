
from logic.embedding import Embedding

em = Embedding(lang='fr')
em.part_embedding(model_name='phoneme_embedding', syllable=False)
em.plot('phoneme_embedding_fr')
em.plot_clusters('phoneme_embedding_fr')
em.get_similarity(model_name='phoneme_embedding')
