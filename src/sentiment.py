import pandas as pd
import numpy as np
import re
from pathlib import Path
import math

# VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# paths 
DATA_RAW  = Path("data/raw")
MODEL_DIR = Path("models/distilbert_sentiment")

# The split index is fixed so train and eval NEVER overlap.
# 5000 train  ·  1000 val  ·  remainder (~43 000) = eval pool
_TRAIN_END = 5000
_VAL_END   = 6000   # indices [5000, 6000) are validation
# evaluate.py samples from indices [6000, …) — never seen during training



# UTILITIES


def clean_review(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_imdb():
    path = DATA_RAW / "imdb" / "IMDB Dataset.csv"
    df   = pd.read_csv(path)
    df['review'] = df['review'].apply(clean_review)
    df['label']  = (df['sentiment'] == 'positive').astype(int)
    print(f"✓ IMDB loaded and cleaned: {df.shape}")
    return df


def get_imdb_splits(df):
    """
    Return (train_df, val_df, eval_df) with guaranteed zero overlap.

    Uses a fixed seed shuffle so splits are reproducible across
    all scripts, but the seed is applied ONCE here — not scattered
    across load_imdb / train / evaluate calls.

    train : rows [0,      _TRAIN_END)  → fine-tuning only
    val   : rows [_TRAIN_END, _VAL_END) → used during training loop
    eval  : rows [_VAL_END,  end)       → evaluate.py samples from here
    """
    shuffled  = df.sample(frac=1, random_state=0).reset_index(drop=True)
    train_df  = shuffled.iloc[:_TRAIN_END].copy()
    val_df    = shuffled.iloc[_TRAIN_END:_VAL_END].copy()
    eval_df   = shuffled.iloc[_VAL_END:].copy()
    print(f"✓ IMDB split → train={len(train_df):,}  "
          f"val={len(val_df):,}  eval={len(eval_df):,}")
    return train_df, val_df, eval_df



# 1. VADER


def build_vader():
    analyzer = SentimentIntensityAnalyzer()
    print("✓ VADER analyzer ready")
    return analyzer


def vader_score(text, analyzer):
    return analyzer.polarity_scores(text)['compound']


def vader_score_list(reviews, analyzer):
    if not reviews:
        return 0.0
    return float(np.mean([vader_score(r, analyzer) for r in reviews]))


def evaluate_vader(df, analyzer, sample=500):
    sample_df = df.sample(n=min(sample, len(df)), random_state=1)
    correct   = sum(
        1 for _, row in sample_df.iterrows()
        if (1 if vader_score(row['review'], analyzer) > 0 else 0)
           == row['label']
    )
    accuracy = correct / len(sample_df)
    print(f"✓ VADER accuracy on {len(sample_df)} IMDB reviews: {accuracy:.1%}")
    return accuracy



# 2. RE-RANKER


def rerank_with_sentiment(hybrid_recs, analyzer,
                           tmdb_df=None,
                           distilbert_model=None,
                           distilbert_tokenizer=None,
                           hybrid_weight=0.70,
                           tmdb_weight=0.15,
                           distilbert_weight=0.15):
    assert abs(hybrid_weight + tmdb_weight + distilbert_weight - 1.0) < 1e-6, \
        "weights must sum to 1.0"

    print(f"\nRe-ranking {len(hybrid_recs)} recommendations "
          f"(hybrid={hybrid_weight}, tmdb={tmdb_weight}, "
          f"distilbert={distilbert_weight})...")

    results = []
    for rec in hybrid_recs:
        title        = rec['title']
        tmdb_score   = 0.5
        overview_text = ''

        if tmdb_df is not None:
            row = tmdb_df[tmdb_df['title'].str.lower() == title.lower()]
            if len(row) == 0:
                clean = re.sub(r'\s*\(\d{4}\)\s*$', '',
                               title).strip().lower()
                row = tmdb_df[tmdb_df['title'].str.lower() == clean]

            if len(row) > 0:
                r          = row.iloc[0]
                vote_avg   = float(r.get('vote_average', 5) or 5)
                vote_count = float(r.get('vote_count',   0) or 0)
                quality    = vote_avg / 10.0
                confidence = min(math.log10(vote_count + 1) / 5.0, 1.0)
                tmdb_score = round(quality * confidence, 4)
                overview_text = str(r.get('overview', '') or '')

        db_score = 0.5
        if overview_text:
            if distilbert_model is not None and distilbert_tokenizer is not None:
                try:
                    db_score = distilbert_score(
                        overview_text, distilbert_model, distilbert_tokenizer
                    )
                except Exception:
                    db_score = 0.5
            else:
                compound = analyzer.polarity_scores(overview_text)['compound']
                db_score = round((compound + 1) / 2, 4)

        final_score = round(
            hybrid_weight     * rec['hybrid_score'] +
            tmdb_weight       * tmdb_score          +
            distilbert_weight * db_score,
            4
        )

        results.append({
            'title':            rec['title'],
            'hybrid_score':     rec['hybrid_score'],
            'tmdb_score':       tmdb_score,
            'distilbert_score': round(db_score, 4),
            'sentiment_score':  round(db_score, 4),
            'final_score':      final_score
        })

    results.sort(key=lambda x: x['final_score'], reverse=True)
    return results



# 3. DISTILBERT

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (DistilBertTokenizerFast,
                          DistilBertForSequenceClassification)
from torch.optim import AdamW


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_distilbert(df, epochs=2, batch_size=16, max_length=256):
    """
    Fine-tune DistilBERT using the fixed train/val split from
    get_imdb_splits().  The eval split is never touched here.
    """
    train_df, val_df, _ = get_imdb_splits(df)   # eval split ignored

    print("\nFine-tuning DistilBERT...")
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | "
          f"Epochs: {epochs} | Batch: {batch_size}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        'distilbert-base-uncased'
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2
    )
    model.to(device)

    train_dataset = IMDBDataset(
        train_df['review'].tolist(),
        train_df['label'].tolist(),
        tokenizer, max_length
    )
    val_dataset = IMDBDataset(
        val_df['review'].tolist(),
        val_df['label'].tolist(),
        tokenizer, max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct    = 0
        total      = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss   = outputs.loss
            logits = outputs.logits

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds       = torch.argmax(logits, dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            if (batch_idx + 1) % 50 == 0:
                print(f"    Epoch {epoch+1} | "
                      f"Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {total_loss/(batch_idx+1):.4f} | "
                      f"Train Acc: {correct/total:.1%}")

        model.eval()
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels         = batch['label'].to(device)

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                preds   = torch.argmax(outputs.logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total
        print(f"\n  ✓ Epoch {epoch+1} complete | Val Accuracy: {val_acc:.1%}\n")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"✓ DistilBERT saved to {MODEL_DIR}")
    return model, tokenizer


def load_distilbert():
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model     = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    print(f"✓ DistilBERT loaded from {MODEL_DIR}")
    return model, tokenizer


def distilbert_score(text, model, tokenizer, max_length=256):
    """Returns probability of positive sentiment (0–1)."""
    device   = next(model.parameters()).device
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    input_ids      = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1)
        return float(probs[0][1])


def evaluate_distilbert(df, model, tokenizer, sample=200):
    """Test on held-out eval split only — never touches train/val rows."""
    _, _, eval_df = get_imdb_splits(df)
    sample_df = eval_df.sample(n=min(sample, len(eval_df)), random_state=2)
    correct   = 0

    for _, row in sample_df.iterrows():
        score     = distilbert_score(row['review'], model, tokenizer)
        predicted = 1 if score > 0.5 else 0
        if predicted == row['label']:
            correct += 1

    accuracy = correct / len(sample_df)
    print(f"✓ DistilBERT accuracy on {len(sample_df)} held-out reviews: {accuracy:.1%}")
    return accuracy

 
# MAIN


if __name__ == "__main__":
    print("=" * 55)
    print("PHASE 4 — Sentiment Analysis")
    print("=" * 55)

    imdb     = load_imdb()
    analyzer = build_vader()

    print("\n--- VADER on sample reviews ---")
    for i in range(3):
        review = imdb['review'].iloc[i]
        score  = vader_score(review, analyzer)
        actual = imdb['sentiment'].iloc[i]
        print(f"\n  Review snippet : {review[:80]}...")
        print(f"  VADER compound : {score:.4f}")
        print(f"  Actual label   : {actual}")
        print(f"  VADER predicted: {'positive' if score > 0 else 'negative'}")

    print("\n--- VADER accuracy ---")
    _, _, eval_df = get_imdb_splits(imdb)
    evaluate_vader(eval_df, analyzer, sample=500)

    print("\n--- DistilBERT fine-tuning ---")
    model, tokenizer = train_distilbert(imdb, epochs=2, batch_size=16)

    print("\n--- DistilBERT accuracy on held-out eval set ---")
    evaluate_distilbert(imdb, model, tokenizer, sample=300)