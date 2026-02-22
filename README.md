# NLP Course Projects &mdash; Fall 2025

A collection of NLP projects completed for the Natural Language Processing course (Fall 2025). The main course project focuses on Dimensional Aspect-Based Sentiment Analysis, complemented by three homework assignments covering authorship identification, POS tagging, and sentiment analysis fine-tuning.

## Repository Structure

```
.
├── project/                              # Course Project
│   ├── dimabsa_experiments.ipynb         # Full experiment suite (14+ experiments)
│   └── dimabsa_submission.ipynb          # Final submission notebook
├── homework/
│   ├── hw1-authorship-identification/
│   │   └── authorship_identification.ipynb
│   ├── hw2-hmm-pos-tagger/
│   │   └── hmm_pos_tagger.ipynb
│   └── hw3-sentiment-finetuning/
│       └── sentiment_finetuning.ipynb
└── README.md
```

---

## Course Project: Dimensional Aspect-Based Sentiment Analysis

**Task:** DimABSA Shared Task &mdash; Track A, Subtask 1 (English)

Predict continuous **valence** (positive/negative) and **arousal** (calm/excited) scores for specific aspects mentioned in laptop and restaurant reviews.

**Result:** Ranked **14th out of 250+ submissions** in Phase 1.

### Approach

- Transformer-based regression models that take review text + aspect term as input and predict valence/arousal scores on a 1&ndash;9 scale.
- Systematically explored **14+ experiment configurations** including:
  - **Pretrained encoders:** BERT-base, RoBERTa-base, DeBERTa-v3-base, VAD-BERT
  - **Architectures:** Simple [CLS] head, deeper MLP head, separate V/A heads, multi-layer [CLS] concatenation, attention pooling
  - **Input formats:** SEP, colon, question, marker, detailed
  - **Loss functions:** MSE, weighted MSE
  - **Other:** Classification with binning, combined regression + classification, ensemble of 3 models
- Hyperparameter search over learning rate, batch size, epochs, and dropout.
- Domain-specific training (laptop vs. restaurant) and combined-domain training with domain flags.
- Evaluation using official RMSE and Pearson correlation metrics.

### Key Results

| Model                       | RMSE   | PCC (Valence) | PCC (Arousal) |
|-----------------------------|--------|---------------|---------------|
| Combined-BERT-Laptop        | 1.0999 | 0.9060        | 0.7356        |
| Combined-BERT-Restaurant    | 1.1438 | 0.8975        | 0.7319        |
| Combined-RoBERTa-Laptop     | 1.1158 | 0.8994        | 0.7170        |
| Combined-RoBERTa-Restaurant | 1.1273 | 0.8962        | 0.7395        |

### Tools

Python, PyTorch, Hugging Face Transformers, Google Colab (A100 GPU)

---

## Homework 1: Authorship Identification

**Task:** Classify song lyrics by artist (21 artists, ~5,500 lyrics from Kaggle).

### Approach

- **Data cleaning:** Regex + semantic similarity filtering (MiniLM) to remove placeholder lyrics.
- **Preprocessing:** Lemmatization and stopword removal via spaCy.
- **Feature engineering:** Bag-of-Words (unigram/bigram), word-level TF-IDF (1&ndash;3 grams), character-level TF-IDF (3&ndash;5 grams), and hybrid TF-IDF (word + char).
- **Models:**
  - Multinomial Naive Bayes (implemented from scratch)
  - Logistic Regression
  - Linear SVM
  - Bidirectional LSTM with GloVe embeddings

### Key Results

| Model                        | Accuracy | Weighted F1 |
|------------------------------|----------|-------------|
| Naive Bayes (from scratch)   | 0.7349   | 0.7320      |
| Logistic Regression (Hybrid) | 0.8012   | 0.7970      |
| **Linear SVM (Hybrid)**      | **0.8229** | **0.8210** |
| BiLSTM + GloVe               | 0.4542   | 0.4460      |

---

## Homework 2: HMM POS Tagger

**Task:** Part-of-speech tagging on Turkish treebank data using a Hidden Markov Model built from scratch.

### Approach

- **Data:** Combined Turkish web + wiki treebanks (~4,850 sentences), split 80/20.
- **HMM implementation (from scratch):**
  - MLE-based transition and emission probabilities
  - Laplace smoothing with rare-word-aware scaling
  - Viterbi algorithm for decoding
  - Unambiguous lemma dictionary for constraint-based prediction
- **Two models:**
  - Model A: All 14 POS tags
  - Model B: Restricted to ADJ, ADV, NOUN, VERB, PUNCT

### Key Results

| Model                    | Accuracy | Weighted F1 |
|--------------------------|----------|-------------|
| Model A (All 14 tags)    | ~0.91    | ~0.91       |
| **Model B (5 tags)**     | **~0.95**| **~0.95**   |

---

## Homework 3: Sentiment Analysis Fine-Tuning

**Task:** Fine-tune three different LLM architectures (encoder-only, decoder-only, encoder-decoder) on Turkish movie reviews for binary sentiment classification.

### Approach

- **Dataset:** 10,660 Turkish movie reviews (balanced positive/negative), split 70/15/15.
- **Models:**
  - **BERTurk** (encoder-only) &mdash; standard fine-tuning with classification head
  - **Qwen3-1.7B** (decoder-only) &mdash; LoRA fine-tuning with prompt-based generation
  - **mT5-base** (encoder-decoder) &mdash; text-to-text fine-tuning
- 3 hyperparameter configurations per model (9 experiments total).

### Key Results

| Model       | Config | Dev F1 | Test F1 |
|-------------|--------|--------|---------|
| **BERTurk** | exp1   | 0.9237 | **0.9206** |
| BERTurk     | exp3   | 0.9193 | 0.9206  |
| Qwen3-1.7B  | exp2   | 0.8981 | 0.8968  |
| Qwen3-1.7B  | exp1   | 0.9056 | 0.8862  |
| mT5-base    | exp2   | 0.9006 | 0.8868  |
| mT5-base    | exp1   | 0.7055 | 0.6840  |

---

## Tools & Technologies

- **Languages:** Python
- **Libraries:** PyTorch, Hugging Face Transformers, PEFT (LoRA), scikit-learn, spaCy, TensorFlow/Keras, NumPy, pandas, matplotlib, seaborn
- **Compute:** Google Colab (NVIDIA A100)

## How to Run

All notebooks are designed to run on **Google Colab**. Each notebook downloads its own dataset and dependencies at the top.

1. Upload the notebook to Google Colab.
2. Select a GPU runtime (A100 recommended for HW3 and the course project).
3. Run all cells sequentially.
