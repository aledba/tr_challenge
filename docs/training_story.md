# Training Story: Legal-Longformer for Procedural Posture Classification

**Thomson Reuters Data Science Challenge**  
*A chronicle of model development, failures, and solutions*

---

## 1. The Starting Point

We began with:
- **18,000 legal documents** with nested sections/paragraphs
- **224 procedural posture labels** (multi-label classification)
- **Severe class imbalance**: top label "On Appeal" appears in 26% of docs; 140 labels have <10 samples
- **Long documents**: median ~2,500 words, some exceeding 100K tokens

Baseline: TF-IDF + Logistic Regression → **F1 Micro: 0.751, F1 Macro: 0.526**

---

## 2. First Attempt: Learning Rate Too HIGH

### Configuration
```python
TrainingConfig(
    learning_rate=2e-5,      # Too aggressive for fine-tuning!
    use_pos_weight=False,    # No class balancing
    batch_size=16,
)
```

### Result
| Metric | Value |
|--------|-------|
| F1 Micro | **0.478** |
| Behavior | Much worse than TF-IDF baseline (0.752) |

**Diagnosis: Catastrophic Forgetting**

With LR=2e-5, the model made aggressive weight updates that destroyed the pretrained legal language understanding. The classification head learned, but the backbone "forgot" how to understand legal text.

---

## 3. Second Attempt: Learning Rate Too LOW

### Configuration
```python
TrainingConfig(
    learning_rate=5e-6,      # Too conservative!
    use_pos_weight=False,    # Still no class balancing
    batch_size=8,
)
```

### Result
| Metric | Value |
|--------|-------|
| F1 Micro | **0.011** |
| Behavior | Model predicts almost nothing! |

**Diagnosis: Shy Model**

With LR=5e-6, the model barely updated its weights. The classification head (initialized randomly) never learned to output confident predictions. All probabilities stayed below 0.5 → almost no positive predictions.

**Plot twist**: We also discovered a **progress bar bug**! The displayed loss was divided by `gradient_accumulation_steps=4`, so we thought loss was 0.08 when it was actually ~0.32. This masked how poorly training was going.

---

## 4. The Fix: `pos_weight` + Middle-Ground LR

### Two Problems, Two Solutions

**Problem 1: Class Imbalance**
- "On Appeal" has 9,197 samples (55% of data)
- Some classes have only 50 samples
- BCELoss sees way more negatives → model learns "when in doubt, predict 0"

**Solution: `pos_weight`**
```python
# Calculate pos_weight: ratio of negatives to positives per class
pos_count = train_labels.sum(axis=0) + 1e-8
neg_count = len(train_labels) - pos_count
pos_weight = neg_count / pos_count
pos_weight = np.clip(pos_weight, 1.0, 50.0)  # Clip to avoid extreme weights

self._loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
```

This makes rare classes "louder"—a missed rare label hurts 10-50x more than a missed common one.

**Problem 2: Learning Rate**
- 2e-5: Too high → catastrophic forgetting
- 5e-6: Too low → model too cautious

**Solution: The Goldilocks LR**
```python
learning_rate = 1e-5  # Not too hot, not too cold
```

### The Learning Rate Journey

| Attempt | LR | F1 Micro | What Happened |
|---------|-----|----------|---------------|
| 1 | `2e-5` | 0.478 | Too HIGH → destroyed pretrained knowledge |
| 2 | `5e-6` | 0.011 | Too LOW → model too shy, barely learned |
| 3 | `1e-5` | **0.628** | Just right! Stable, steady improvement |

### Why 1e-5 Works

Legal-Longformer is already pre-trained on legal text. Fine-tuning needs a gentle touch:
- Preserves the learned legal representations
- Allows the classification head to adapt gradually
- Prevents catastrophic forgetting while still learning

---

## 5. Empty Text Discovery

### The Problem

During validation, we found **3 documents** with essentially empty text (<10 words after extraction). These caused:
- NaN losses during training
- Cryptic errors in evaluation
- Misleading metrics

### The Fix

```python
# Filter out empty or minimal text documents (< 10 words)
MIN_WORDS = 10
valid_text_mask = [len(t.split()) >= MIN_WORDS for t in texts]
n_filtered = sum(1 for m in valid_text_mask if not m)
if n_filtered > 0:
    texts = [t for t, m in zip(texts, valid_text_mask) if m]
    labels = [l for l, m in zip(labels, valid_text_mask) if m]
    logger.info(f"Filtered {n_filtered} documents with < {MIN_WORDS} words")
```

---

## 6. Threshold Optimization

### Default Threshold (0.5)
| Metric | Value |
|--------|-------|
| F1 Micro | 0.637 |

### Optimized Threshold (0.6)
| Metric | Value |
|--------|-------|
| F1 Micro | **0.701** |

Raising the threshold reduced false positives for rare classes while maintaining recall on frequent ones. The +6.4pp gain came "for free" via calibration.

---

## 7. Final Configuration 

```python
TrainingConfig(
    batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size: 16
    learning_rate=1e-5,             # Conservative for fine-tuning
    num_epochs=5,
    warmup_ratio=0.1,               # 10% warmup
    weight_decay=0.01,
    max_grad_norm=1.0,
    early_stopping_patience=2,
    use_pos_weight=True,            # THE KEY: class balancing
)

DataPreparer(
    min_label_count=50,             # Only classes with ≥50 samples (41 classes)
    # MIN_WORDS=10 filter applied internally
)
```

---

## 8. Final Results 

### Training Progress (with `pos_weight` + LR=1e-5)

| Epoch | Train Loss | Val Loss | Val F1 Micro |
|-------|------------|----------|--------------|
| 1 | 0.936 | 0.823 | 0.396 |
| 2 | 0.680 | 0.539 | 0.454 |
| 3 | 0.487 | 0.426 | 0.545 |
| **4** | **0.392** | **0.385** | **0.628** ← Best |
| 5 | 0.332 | 0.353 | 0.624 |

Model auto-restored to epoch 4 (best validation F1).

### Final Comparison (Test Set)

*Thresholds optimized on validation set, evaluated on held-out test set (proper methodology)*

| Metric | TF-IDF | LF (t=0.5) | LF (t=0.6) | LF Per-Class |
|--------|--------|------------|------------|--------------|
| **F1 Micro** | 0.752 | 0.637 | 0.701 | **0.770** |
| **F1 Macro** | 0.569 | 0.458 | 0.528 | **0.585** |
| Precision | 0.744 | 0.483 | 0.586 | 0.708 |
| Recall | 0.760 | 0.935 | 0.872 | 0.843 |

**Gap to TF-IDF (F1 Micro): +1.8%**

Key insight: Per-class threshold optimization gave the biggest gains, outperforming TF-IDF on both Micro and Macro F1. The validation/test split ensures no data leakage in threshold tuning.

### Model Card
- **Architecture**: `lexlms/legal-longformer-base` (149M params)
- **Max length**: 4,096 tokens
- **Training**: 5 epochs (~13 hours total) on Apple MPS
- **Best epoch**: 4
- **Checkpoint**: 594 MB

---

## 9. Lessons Learned

1. **Class imbalance is deadly in multi-label**: Without `pos_weight`, the model collapsed to predicting only the majority class
2. **Fine-tuning LR is Goldilocks**: 2e-5 (too hot) → 5e-6 (too cold) → **1e-5** (just right)
3. **Trust but verify**: Progress bar bug showed loss/4 — always validate metrics independently
4. **Data quality matters**: 3 empty docs caused disproportionate debugging time
5. **Threshold tuning is free F1**: +6.4pp from 0.5 → 0.6
6. **Test suite is essential**: 44 tests caught edge cases before they became production bugs

---

## 10. What's Next

Potential improvements:
- **Macro-F1 focus**: Current model favors frequent classes
- **Hierarchical loss**: Exploit ontology structure (Stage > Motion > Characteristic)
- **Label attention**: LAMT_MLC-style mechanism for rare classes
- **Longer context**: Legal-LED (16K tokens) for full document coverage

---

*Document created: January 2026*
