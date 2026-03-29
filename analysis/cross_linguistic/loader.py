"""
Utility module for loading embeddings and frequency data for FR-ES cross-linguistic analysis.
"""

from pathlib import Path
from gensim.models import Word2Vec
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "data" / "embedding" / "models"
FREQ_DIR = BASE_DIR / "data" / "embedding" / "frequency"

# Model paths: FR doesn't use _wiki_ infix, ES does
MODEL_PATHS = {
    "phoneme": {
        "es": MODELS_DIR / "phoneme_embedding_wiki_es.model",
        "fr": MODELS_DIR / "phoneme_embedding_fr.model",
    },
    "syllable": {
        "es": MODELS_DIR / "syllable_embedding_wiki_es.model",
        "fr": MODELS_DIR / "syllable_embedding_fr.model",
    },
    "word": {
        "es": MODELS_DIR / "word_embedding_wiki_es.model",
        "fr": MODELS_DIR / "word_embedding_fr.model",
    },
}

FREQ_PATHS = {
    "phoneme": {
        "es": FREQ_DIR / "frequency_phoneme_embedding_wiki_es.csv",
        "fr": FREQ_DIR / "frequency_phoneme_embedding_fr.csv",
    },
    "syllable": {
        "es": FREQ_DIR / "frequency_syllable_embedding_wiki_es.csv",
        "fr": FREQ_DIR / "frequency_syllable_embedding_fr.csv",
    },
    "word": {
        "es": FREQ_DIR / "frequency_word_embedding_wiki_es.csv",
        "fr": FREQ_DIR / "frequency_word_embedding_fr.csv",
    },
}


def load_model(level: str, lang: str) -> Word2Vec:
    path = MODEL_PATHS[level][lang]
    return Word2Vec.load(str(path))


def load_frequency(level: str, lang: str) -> pd.DataFrame:
    path = FREQ_PATHS[level][lang]
    return pd.read_csv(path, sep=";")


# Phoneme inventories: actual phonemes of each language.
# Spanish: RAE standard (~24 phonemes, Epitran spa-Latn output)
# French: standard metropolitan French (~36 phonemes, Epitran fra-Latn output)
PHONEME_INVENTORY = {
    "es": {
        # Vowels
        "a", "e", "i", "o", "u",
        # Plosives
        "p", "b", "t", "d", "k", "ɡ",
        # Fricatives
        "f", "s", "x", "θ",
        # Affricates
        "t͡ʃ",
        # Nasals
        "m", "n", "ɲ",
        # Laterals
        "l", "ʎ",
        # Rhotics
        "r", "ɾ",
        # Glides
        "j", "w",
        # Allophones commonly produced by Epitran
        "β", "ð", "ɣ",  # lenited plosives
        "ŋ",             # allophone of /n/ before velars
    },
    "fr": {
        # Oral vowels
        "a", "e", "ɛ", "i", "o", "ɔ", "u", "y", "ə", "ø", "œ",
        # Nasal vowels
        "ɑ̃", "ɛ̃", "ɔ̃", "œ̃",
        # Plosives
        "p", "b", "t", "d", "k", "ɡ",
        # Fricatives
        "f", "v", "s", "z", "ʃ", "ʒ",
        # Nasals
        "m", "n", "ɲ", "ŋ",
        # Liquids
        "l", "ʁ",
        # Glides
        "j", "w", "ɥ",
    },
}


def get_vocab(model: Word2Vec, lang: str = None) -> set:
    """Get model vocabulary, optionally filtered to the language's phoneme inventory.

    For phoneme-level models, pass `lang` to restrict to real phonemes.
    """
    vocab = set(model.wv.key_to_index.keys())
    if lang and lang in PHONEME_INVENTORY:
        vocab = vocab & PHONEME_INVENTORY[lang]
    return vocab


def get_vectors(model: Word2Vec) -> np.ndarray:
    return model.wv.vectors


def get_shared_tokens(model_a: Word2Vec, model_b: Word2Vec,
                      lang_a: str = None, lang_b: str = None) -> set:
    return get_vocab(model_a, lang_a) & get_vocab(model_b, lang_b)


# Phonological features for each phoneme.
# Keys: phoneme -> dict of features
# Used for evaluating whether embedding clusters recover phonological categories.
PHONEME_FEATURES = {
    # --- Shared (ES + FR) ---
    "a":  {"type": "vowel", "height": "open",      "backness": "central", "rounded": False, "nasal": False},
    "e":  {"type": "vowel", "height": "close-mid",  "backness": "front",   "rounded": False, "nasal": False},
    "i":  {"type": "vowel", "height": "close",      "backness": "front",   "rounded": False, "nasal": False},
    "o":  {"type": "vowel", "height": "close-mid",  "backness": "back",    "rounded": True,  "nasal": False},
    "u":  {"type": "vowel", "height": "close",      "backness": "back",    "rounded": True,  "nasal": False},
    "p":  {"type": "consonant", "manner": "plosive",    "place": "bilabial",     "voiced": False},
    "b":  {"type": "consonant", "manner": "plosive",    "place": "bilabial",     "voiced": True},
    "t":  {"type": "consonant", "manner": "plosive",    "place": "alveolar",     "voiced": False},
    "d":  {"type": "consonant", "manner": "plosive",    "place": "alveolar",     "voiced": True},
    "k":  {"type": "consonant", "manner": "plosive",    "place": "velar",        "voiced": False},
    "ɡ":  {"type": "consonant", "manner": "plosive",    "place": "velar",        "voiced": True},
    "f":  {"type": "consonant", "manner": "fricative",  "place": "labiodental",  "voiced": False},
    "s":  {"type": "consonant", "manner": "fricative",  "place": "alveolar",     "voiced": False},
    "m":  {"type": "consonant", "manner": "nasal",      "place": "bilabial",     "voiced": True},
    "n":  {"type": "consonant", "manner": "nasal",      "place": "alveolar",     "voiced": True},
    "ɲ":  {"type": "consonant", "manner": "nasal",      "place": "palatal",      "voiced": True},
    "ŋ":  {"type": "consonant", "manner": "nasal",      "place": "velar",        "voiced": True},
    "l":  {"type": "consonant", "manner": "lateral",    "place": "alveolar",     "voiced": True},
    "j":  {"type": "consonant", "manner": "approximant","place": "palatal",      "voiced": True},
    "w":  {"type": "consonant", "manner": "approximant","place": "labial-velar", "voiced": True},
    # --- ES only ---
    "θ":  {"type": "consonant", "manner": "fricative",  "place": "dental",       "voiced": False},
    "x":  {"type": "consonant", "manner": "fricative",  "place": "velar",        "voiced": False},
    "t͡ʃ": {"type": "consonant", "manner": "affricate",  "place": "postalveolar", "voiced": False},
    "r":  {"type": "consonant", "manner": "trill",      "place": "alveolar",     "voiced": True},
    "ɾ":  {"type": "consonant", "manner": "tap",        "place": "alveolar",     "voiced": True},
    "ʎ":  {"type": "consonant", "manner": "lateral",    "place": "palatal",      "voiced": True},
    "β":  {"type": "consonant", "manner": "fricative",  "place": "bilabial",     "voiced": True},
    "ð":  {"type": "consonant", "manner": "fricative",  "place": "dental",       "voiced": True},
    "ɣ":  {"type": "consonant", "manner": "fricative",  "place": "velar",        "voiced": True},
    "ʝ":  {"type": "consonant", "manner": "fricative",  "place": "palatal",      "voiced": True},
    # --- FR only ---
    "ɛ":  {"type": "vowel", "height": "open-mid",   "backness": "front",   "rounded": False, "nasal": False},
    "ɔ":  {"type": "vowel", "height": "open-mid",   "backness": "back",    "rounded": True,  "nasal": False},
    "y":  {"type": "vowel", "height": "close",      "backness": "front",   "rounded": True,  "nasal": False},
    "ø":  {"type": "vowel", "height": "close-mid",  "backness": "front",   "rounded": True,  "nasal": False},
    "œ":  {"type": "vowel", "height": "open-mid",   "backness": "front",   "rounded": True,  "nasal": False},
    "ə":  {"type": "vowel", "height": "mid",        "backness": "central", "rounded": False, "nasal": False},
    "ɑ̃":  {"type": "vowel", "height": "open",       "backness": "back",    "rounded": False, "nasal": True},
    "ɛ̃":  {"type": "vowel", "height": "open-mid",   "backness": "front",   "rounded": False, "nasal": True},
    "ɔ̃":  {"type": "vowel", "height": "open-mid",   "backness": "back",    "rounded": True,  "nasal": True},
    "œ̃":  {"type": "vowel", "height": "open-mid",   "backness": "front",   "rounded": True,  "nasal": True},
    "v":  {"type": "consonant", "manner": "fricative",  "place": "labiodental",  "voiced": True},
    "z":  {"type": "consonant", "manner": "fricative",  "place": "alveolar",     "voiced": True},
    "ʃ":  {"type": "consonant", "manner": "fricative",  "place": "postalveolar", "voiced": False},
    "ʒ":  {"type": "consonant", "manner": "fricative",  "place": "postalveolar", "voiced": True},
    "ʁ":  {"type": "consonant", "manner": "fricative",  "place": "uvular",       "voiced": True},
    "ɥ":  {"type": "consonant", "manner": "approximant","place": "labial-palatal","voiced": True},
}


def get_phoneme_features(lang: str) -> pd.DataFrame:
    """Return a DataFrame of phonological features for the given language's inventory."""
    rows = []
    for ph in sorted(PHONEME_INVENTORY[lang]):
        feats = PHONEME_FEATURES.get(ph, {})
        rows.append({"phoneme": ph, **feats})
    return pd.DataFrame(rows).set_index("phoneme")


def load_all_models():
    """Load all FR and ES models for the three levels."""
    models = {}
    for level in ("phoneme", "syllable", "word"):
        models[level] = {}
        for lang in ("es", "fr"):
            models[level][lang] = load_model(level, lang)
    return models


def load_all_frequencies():
    """Load all FR and ES frequency dataframes."""
    freqs = {}
    for level in ("phoneme", "syllable", "word"):
        freqs[level] = {}
        for lang in ("es", "fr"):
            freqs[level][lang] = load_frequency(level, lang)
    return freqs
