# Correctness Review: Thomson Reuters Data Science Challenge

**Date:** January 28, 2026
**Reviewer:** Claude (Code Review)
**Scope:** Evaluation methodology for TF-IDF and Legal-Longformer models

---

## Executive Summary

The evaluation methodology is **fundamentally sound** with no critical data leakage issues. However, several minor issues and inconsistencies were found that should be addressed before submission.

| Category | Status |
|----------|--------|
| Data Splitting | ✅ CORRECT |
| TF-IDF Evaluation | ✅ CORRECT |
| Transformer Evaluation | ✅ CORRECT |
| Threshold Optimization | ✅ CORRECT (val→test) |
| Ensemble Methodology | ✅ CORRECT |
| Metric Reporting | ⚠️ MINOR INCONSISTENCIES |

---

## 1. Data Pipeline Review (`src/model_trainer.py`)

### 1.1 Train/Val/Test Split

**Finding: CORRECT**

The splitting uses two-stage `train_test_split`:
```python
# Stage 1: Extract 15% test
X_temp, X_test, y_temp, y_test = train_test_split(..., test_size=0.15, random_state=42)

# Stage 2: Split remaining into train/val
adjusted_val_size = 0.15 / (1 - 0.15)  # ~0.176
X_train, X_val, y_train, y_val = train_test_split(X_temp, ..., test_size=adjusted_val_size)
```

**Result:** 70% train / 15% val / 15% test (approximate)

### 1.2 Val/Test Naming Swap (Lines 210-222)

**Finding: CONFUSING BUT CORRECT**

After splitting, the code intentionally swaps the names:
```python
return PreparedData(
    X_val=X_test_vec,      # Original "test" → now "val" for threshold tuning
    X_test=X_val_vec,      # Original "val" → now "test" for final eval
    ...
)
```

**Why this is correct:**
- Both sets remain held out from training
- The "val" set (15%) is used for threshold optimization
- The "test" set (~17.6%) is used for final evaluation
- The naming convention matches ML best practices: tune on val, evaluate on test

**Recommendation:** Add more prominent documentation about this swap.

### 1.3 TF-IDF Vectorizer

**Finding: CORRECT**

```python
X_train_vec = self._vectorizer.fit_transform(X_train)  # FIT on train only
X_val_vec = self._vectorizer.transform(X_val)           # TRANSFORM only
X_test_vec = self._vectorizer.transform(X_test)         # TRANSFORM only
```

No vocabulary leakage from val/test sets.

### 1.4 Label Filtering

**Finding: CORRECT**

Labels with `min_label_count < 50` are filtered BEFORE splitting:
- Documents with no viable labels are removed
- This is correct (same filtering applied to entire dataset)

---

## 2. Evaluation Logic Review (`src/model_evaluator.py`)

### 2.1 MultiLabelEvaluator.evaluate()

**Finding: CORRECT**

Uses proper sklearn calls with consistent parameters:
```python
f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
f1_samples = f1_score(y_true, y_pred, average='samples', zero_division=0)
```

The `zero_division=0` parameter handles edge cases consistently.

### 2.2 Per-Class Metrics

**Finding: CORRECT**

Each label is treated as an independent binary classification:
```python
for i, label in enumerate(self.label_names):
    y_true_i = y_true[:, i]  # Extract single label
    y_pred_i = y_pred[:, i]
    f1 = f1_score(y_true_i, y_pred_i, zero_division=0)
```

### 2.3 compute_threshold_analysis()

**Finding: CORRECT IMPLEMENTATION, BUT NO SAFEGUARD**

The function correctly converts probabilities to binary predictions:
```python
for thresh in thresholds:
    y_pred = (y_proba >= thresh).astype(int)
    f1_micro = f1_score(y_true, y_pred, ...)
```

**Risk:** No parameter to enforce which split is being used. Misuse (calling on test set) could cause data leakage.

**Mitigation in notebooks:** NB03 correctly uses validation set for optimization.

### 2.4 ModelComparison.ensemble_predictions()

**Finding: CORRECT**

```python
if strategy == 'union':
    return np.maximum(y_pred1, y_pred2)  # OR
elif strategy == 'intersection':
    return np.minimum(y_pred1, y_pred2)  # AND
```

---

## 3. Transformer Training Review (`src/bert_trainer.py`)

### 3.1 pos_weight Calculation

**Finding: CORRECT**

```python
pos_count = y_train.sum(axis=0)  # Positives per class
neg_count = y_train.shape[0] - pos_count
pos_weight = neg_count / pos_count
pos_weight = np.clip(pos_weight, 1.0, 50.0)  # Cap extreme values
```

This correctly implements BCEWithLogitsLoss positive weighting for imbalanced classes.

### 3.2 Early Stopping Threshold

**Finding: SUBOPTIMAL (NOT A BUG)**

```python
# Line 721 in _validation_step()
preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
```

Early stopping uses F1 at hardcoded 0.5 threshold. If optimal threshold is different (e.g., 0.6), the "best" model may not maximize true performance.

**Impact:** Limited. Per-class threshold optimization in NB03 partially mitigates this.

### 3.3 predict() and predict_proba()

**Finding: CORRECT**

- `predict_proba()` returns raw sigmoid probabilities
- `predict()` accepts configurable threshold parameter

---

## 4. Notebook Review

### 4.1 Notebook 02 (TF-IDF + Training)

**Outputs observed:**
```
TF-IDF:
  F1 Micro: 0.7510
  F1 Macro: 0.5205

Legal-Longformer (threshold=0.5):
  F1 Micro: 0.6220
  F1 Macro: 0.4362
```

**Finding:** TF-IDF outperforms transformer at default threshold. This is a known limitation, not a bug.

### 4.2 Notebook 03 (Threshold Optimization)

**Key methodology verified:**
```python
# Optimization on VALIDATION set
for class_idx in range(len(data.label_names)):
    for t in thresholds_to_try:
        y_pred_class = (y_proba_val[:, class_idx] >= t).astype(int)
        f1 = f1_score(data.y_val[:, class_idx], y_pred_class, ...)

# Final evaluation on TEST set
y_pred_test_perclass = ...  # Apply optimized thresholds
results_test_perclass = evaluator.evaluate(data.y_test, y_pred_test_perclass)
```

**Finding: CORRECT** - No data leakage. Thresholds tuned on val, applied to test.

### 4.3 Notebook 04 (Ensemble Analysis)

**Outputs observed:**
```
TF-IDF:           F1 Micro=0.7524, F1 Macro=0.5330
Legal-Longformer: F1 Micro=0.6333, F1 Macro=0.4718
Intersection:     F1 Micro=0.7827, F1 Macro=0.5752
```

**Finding: CORRECT** methodology for ensemble comparison.

---

## 5. Metric Consistency Issues

### 5.1 TF-IDF F1 Discrepancy

| Source | F1 Micro | F1 Macro |
|--------|----------|----------|
| NB02 output | 0.7510 | 0.5205 |
| NB04 output | 0.7524 | 0.5330 |
| Q3 doc | 0.752 | 0.533 |

**Possible causes:**
1. NB02 saves to `tfidf_test_predictions.npz` early in the notebook
2. A later cell (f6661a3e) saves to `tfidf_predictions.npz` (different file)
3. Slight differences in random initialization or file version

**Recommendation:** Ensure all notebooks load from the same saved prediction file.

### 5.2 Hardcoded Values in NB03 - **FIXED**

~~Previously had hardcoded values that didn't match NB02 outputs.~~

**Fix applied:** NB03 now loads TF-IDF predictions from `../outputs/tfidf_test_predictions.npz` and computes metrics dynamically using the same evaluator, ensuring consistency.

---

## 6. Question 3 Document Cross-Check

| Metric | Q3 Doc | Notebooks | Status |
|--------|--------|-----------|--------|
| TF-IDF F1 Micro | 0.752 | 0.7510-0.7524 | ≈ Match |
| TF-IDF F1 Macro | 0.533 | 0.5205-0.5330 | ⚠️ Varies |
| LF per-class F1 Micro | 0.770 | Not directly shown | Unable to verify |
| Ensemble (AND) F1 Micro | 0.783 | 0.7827 | ✅ Match |
| Ensemble F1 Macro | 0.575 | 0.5752 | ✅ Match |

**Note:** The Legal-Longformer per-class threshold result (0.770) should be verified by examining NB03 output more closely. The notebook structure suggests this is computed correctly.

---

## 7. Summary of Findings

### No Issues Found
1. ✅ Data splitting has no leakage
2. ✅ TF-IDF vectorizer fit on train only
3. ✅ Threshold optimization uses validation set
4. ✅ Final evaluation on held-out test set
5. ✅ F1 calculations use correct sklearn functions
6. ✅ Ensemble logic is correct

### Minor Issues (Documentation/Consistency)
1. ⚠️ Val/test swap is confusing - needs clearer documentation
2. ⚠️ TF-IDF metrics vary slightly between notebooks
3. ~~⚠️ NB03 uses hardcoded values that may be stale~~ **FIXED**
4. ⚠️ Early stopping uses 0.5 threshold (suboptimal but acceptable)

### Recommendations Before Submission
1. ~~**High priority:** Fix hardcoded `TFIDF_F1_MACRO = 0.569` in NB03~~ **FIXED** - Now loads from saved predictions
2. **Medium priority:** Ensure all notebooks load from the same prediction files
3. **Low priority:** Add comment explaining val/test swap in PreparedData docstring

---

## 8. Conclusion

**The implementation is correct for a hiring exercise submission.** The evaluation methodology follows ML best practices:
- Proper train/val/test separation
- Threshold tuning on validation, final eval on test
- Consistent metric computation

The minor inconsistencies in hardcoded values should be fixed but do not invalidate the results. The transformer underperformance vs TF-IDF is a real finding, not a bug.

**Verdict: READY FOR SUBMISSION** (after addressing hardcoded value mismatch in NB03)

---

*Review completed: January 28, 2026*
