# Project: Cross-Lingual Transfer Learning for Low-Resource Language Translation


# 🌍 Phoneme-Aware Cross-Lingual Transfer Learning for Low-Resource Language Translation (Unsupervised)

This project implements an end-to-end unsupervised pipeline to perform **speech-to-text translation for low-resource languages (LRLs)** by leveraging **cross-lingual transfer learning** from high-resource languages (HRLs). It uses **phoneme-like unit discovery**, **self-supervised speech embeddings**, and a **universal decoder** trained only on HRL data.

---

## Pipeline Overview

```mermaid
flowchart TD
  A[Raw LRL Speech] --> B[Self-Supervised Feature Extraction]
  B --> C[Pseudo-Phoneme Discovery]
  C --> D[Phoneme Space Alignment]
  D --> E[Universal Decoder (Translation)]
  E --> F[HRL Text Output]
```

---

##  Repository Structure

```
su_project/
├── data_preparation.ipynb         # Download and preprocess speech data
├── feature_extraction.ipynb       # Extract embeddings using wav2vec2
├── unit_discovery.ipynb           # Cluster embeddings into pseudo-phonemes
├── model_training.ipynb           # Train universal phoneme-to-text decoder
├── inference.ipynb                # Translate unseen LRL speech
├── evaluation.ipynb               # Evaluate model outputs (BLEU, WER, etc.)
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

---

## Datasets

### ✅ High-Resource Languages (HRLs)
- [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0)


### Low-Resource Languages (LRLs)
- [Common Voice (LRL subsets)](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0)

---

## Model Components

| Stage                | Description                                                    |
|----------------------|----------------------------------------------------------------|
| Feature Extraction   | Self-supervised models like `wav2vec2` |
| Unit Discovery       | Clustering with `KMeans`               |
| Phoneme Alignment    | IPA projection using `Epitran`         |
| Universal Decoder    | Transformer model for phoneme-to-text translation              |
| Adaptation           | Self-training, back-translation, adapters|

---

## Setup

### Install requirements

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch torchaudio transformers datasets sacrebleu jiwer editdistance sentencepiece
```

---

## Usage Instructions

> Each notebook is a standalone step in the pipeline.

1. **Prepare Data**  
   `data_preparation.ipynb` downloads Common Voice or Wilderness speech data and prepares splits.

2. **Extract Features**  
   `feature_extraction.ipynb` extracts multilingual embeddings using pre-trained models (e.g., wav2vec2 XLS-R or Whisper).

3. **Unit Discovery**  
   `unit_discovery.ipynb` clusters the extracted embeddings into discrete phoneme-like units using k-means or vq-wav2vec.

4. **Train Universal Decoder**  
   `model_training.ipynb` trains a Transformer decoder on HRL phoneme-to-text pairs using aligned IPA embeddings.

5. **Inference**  
   `inference.ipynb` performs zero-shot translation of LRL speech by passing pseudo-phonemes through the trained decoder.

6. **Evaluation**  
   `evaluation.ipynb` evaluates translation performance using BLEU, CHRF, WER, and optionally PER.

---

## 📊 Evaluation Metrics

| Metric | Description                                |
|--------|--------------------------------------------|
| BLEU   | Translation quality                        |
| CHRF   | Character-level F-score                    |
| WER    | Word Error Rate (optional)                 |
| PER    | Phoneme Error Rate (optional)              |

---

## Future Work
- Support more LRLs 
- Integrate alignment via `CTC` and `Wav2Vec2 + IPA`
- Expand decoder with multilingual text generation capabilities

---

## Acknowledgements
- [Facebook AI Research](https://ai.facebook.com/) for wav2vec2
- [Common Voice](https://commonvoice.mozilla.org/)

---
