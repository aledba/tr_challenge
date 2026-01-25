# Thomson Reuters Data Science Challenge

## Project Overview

This is a **hiring exercise** for Thomson Reuters Labs. The task is to build an exploratory model for **procedural posture classification** of judicial opinions.

**Key constraint**: This is about *characterizing performance*, not maximizing it. Clear documentation and feasibility analysis matter more than SOTA results.

---

## Challenge Questions

### Question 1: Data Exploration ✅ COMPLETE
- Load JSON Lines data
- Report: documents, postures, paragraphs
- Describe data relevant to modeling

**Deliverable**: [notebooks/01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb)

### Question 2: Modeling 🔄 IN PROGRESS
- Build a model for procedural posture automation
- Analyze strengths/weaknesses
- **Make feasibility recommendation** comparing to human annotator agreement (κ = 0.63–0.93)

**Deliverables**:
- [notebooks/02_modeling.ipynb](notebooks/02_modeling.ipynb) - TF-IDF baseline DONE
- Legal-BERT fine-tuning - **TODO on powerful machine**

### Question 3: Next Steps ⏳ NOT STARTED
- Either (a) additional experiments OR (b) production planning
- Accessible to technical and non-technical audiences
- Identify potential challenges

---

## Current Progress

### Data Statistics
- **18,000 documents**
- **224 unique postures** (multi-label: 49.8% of docs have multiple)
- **79,000+ paragraphs**
- Nested structure: `sections[].paragraphs[]`

### Label Distribution Tiers
| Tier | Count | Labels | Notes |
|------|-------|--------|-------|
| Common (≥100) | 27 | 91.5% coverage | Good for ML |
| Viable (≥50) | 41 | Used in modeling | min_label_count=50 |
| Rare (<10) | 140 | — | Excluded |

### Baseline Results (TF-IDF + Logistic Regression)
```
F1 Micro:    0.752
F1 Macro:    0.526
F1 Weighted: 0.798

Feasibility vs human κ (0.63–0.93):
- Automatable (F1 ≥ 0.63):  14/41 postures
- High confidence (≥0.93):   1/41 postures
- Needs review (0.50-0.63):  7/41 postures
```

**Key finding**: Performance strongly correlates with sample size. Postures with 500+ samples achieve F1 > 0.92.

---

## Next Step: Legal-BERT Fine-Tuning

### Why Legal-BERT?
- Domain-specific pretraining on legal text
- Model: `nlpaueb/legal-bert-base-uncased`
- Expected improvement: +5-15% F1 over TF-IDF

### Hardware Requirements
| Component | Needed |
|-----------|--------|
| GPU VRAM | 16GB+ recommended |
| RAM | 32GB+ |
| Model size | ~110M params |

### Implementation Plan

1. **Create `src/bert_trainer.py`** with:
   - `LegalBertDataset` - PyTorch Dataset for text + multi-label
   - `LegalBertTrainer` - Training loop with early stopping
   - Uses `BCEWithLogitsLoss` for multi-label

2. **Add cells to `02_modeling.ipynb`**:
   - Load Legal-BERT tokenizer and model
   - Add classification head (768 → 41 labels)
   - Train with gradient accumulation if needed
   - Compare to baseline

3. **Key hyperparameters**:
   ```python
   max_length = 512  # BERT limit
   batch_size = 8-16  # depending on VRAM
   learning_rate = 2e-5
   epochs = 3-5
   ```

4. **Document handling** (docs are long!):
   - Option A: Truncate to first 512 tokens
   - Option B: Chunk and aggregate predictions
   - Start with Option A for simplicity

---

## Key Documentation

### Legal Posture Ontology
**File**: [docs/legal_posture_ontology.md](docs/legal_posture_ontology.md)

This documents the **hierarchical taxonomy** of procedural postures with Mermaid UML diagrams:

- **Categories**: Motion Practice, Trial Stage, Appellate Stage, etc.
- **IS-A relationships**: e.g., "MTD for Lack of SMJ" IS-A "Motion to Dismiss"
- **DEPENDS-ON**: e.g., "Appellate Review (Criminal)" DEPENDS-ON "On Appeal"

**Critical insight from [notebooks/ontology_verification.ipynb](notebooks/ontology_verification.ipynb)**:
> Low co-occurrence of IS-A labels (3-13%) reflects **sparse labeling practice** (labelers use most-specific label only), NOT invalid ontology. The hierarchy is ontologically correct.

This matters for:
- Understanding label relationships
- Potential hierarchical loss functions
- Error analysis (confusing related postures is less severe)

### Challenge Requirements
**File**: [docs/challenge_description.txt](docs/challenge_description.txt)

OCR'd from the PDF. Contains exact wording of all 3 questions.

---

## Project Structure

```
TR_hiring_exercise/
├── CLAUDE.md              # This file
├── data/
│   ├── TRDataChallenge2023.txt   # JSON Lines (18k docs)
│   └── posture_taxonomy.json     # Label distribution
├── docs/
│   ├── legal_posture_ontology.md # Taxonomy with UML diagrams
│   ├── challenge_description.txt # OCR'd requirements
│   └── data_analysis.md
├── notebooks/
│   ├── 01_data_exploration.ipynb # Q1 - COMPLETE
│   ├── 02_modeling.ipynb         # Q2 - Baseline done, BERT TODO
│   └── ontology_verification.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # DataLoader class
│   ├── data_analyzer.py    # DatasetAnalyzer, PostureTaxonomy
│   ├── model_trainer.py    # DataPreparer, BaselineTrainer
│   ├── model_evaluator.py  # MultiLabelEvaluator
│   └── visualization.py
└── outputs/
```

---

## Code Patterns

All logic lives in `src/` modules. Notebooks are clean and concise, just calling module functions.

```python
# Example usage pattern
from src.data_loader import DataLoader
from src.model_trainer import DataPreparer, BaselineTrainer
from src.model_evaluator import MultiLabelEvaluator

loader = DataLoader('../data/TRDataChallenge2023.txt')
preparer = DataPreparer(loader, min_label_count=50)
data = preparer.prepare(max_features=10000)

trainer = BaselineTrainer()
trainer.train(data.X_train, data.y_train)

evaluator = MultiLabelEvaluator(data.label_names)
results = evaluator.evaluate(data.y_test, trainer.predict(data.X_test))
```

---

## Environment

```bash
conda activate tr_challenge
# Python 3.11
# Key packages: pandas, numpy, scikit-learn, matplotlib, seaborn

# For Legal-BERT, install:
pip install torch transformers
```

---

## Immediate TODO

1. **On powerful machine**: Add Legal-BERT to `02_modeling.ipynb`
2. **Compare** baseline vs BERT performance
3. **Update feasibility analysis** with BERT results
4. **Write Question 3** response (next steps)
5. **Package** for submission (zip without data)
