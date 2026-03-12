# Phonetic Affect Intensity Multilingual

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4301068.svg)](https://doi.org/10.5281/zenodo.4301068)

The main objective of the software is to analyze the contribution to the prediction of the detection of polarity in
microblogging sources of the phonetic elements (phonestheme) and emotional elements other than valence (arousal / dominance).

## French Embeddings (from scratch)

Use Python 3.8 for this repository.

```powershell
pip install -r requirements.txt
python -m spacy download fr_core_news_md
```

Run the French embedding scripts to train:

```powershell
python run\generate_word_embedding_fr.py
python run\generate_syllable_embedding_fr.py
python run\generate_phoneme_embedding_fr.py
```

Expected generated model files:
- `data\embedding\models\word_embedding_fr.model`
- `data\embedding\models\syllable_embedding_fr.model`
- `data\embedding\models\phoneme_embedding_fr.model`

## Team

- [Edwin Puertas, PhDc](epuerta@utb.edu.co)
- [Jorge Andres Alvarado Valencia, PhD](jorge.alvarado@javeriana.edu.co)
