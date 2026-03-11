
from logic.embedding import Embedding

em = Embedding(lang='fr')
em.part_embedding(model_name='syllable_embedding')
em.plot('syllable_embedding_fr')
em.plot_clusters('syllable_embedding_fr')
em.get_similarity(model_name='syllable_embedding')
