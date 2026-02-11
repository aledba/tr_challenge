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

### Question 2: Modeling ✅ COMPLETE
- Build a model for procedural posture automation
- Analyze strengths/weaknesses
- **Make feasibility recommendation** comparing to human annotator agreement (κ = 0.63–0.93)

**Deliverables**:
- [notebooks/02_modeling.ipynb](notebooks/02_modeling.ipynb) - TF-IDF baseline + Legal-Longformer hybrid
- [src/bert_trainer.py](src/bert_trainer.py) - Transformer training module

### Question 3: Next Steps ⏳ NOT STARTED
- Either (a) additional experiments OR (b) production planning
- Accessible to technical and non-technical audiences
- Identify potential challenges

---

## Current Progress

### Data Statistics
- **18,000 documents**
- **224 unique postures** (multi-label: 49.8% of docs have multiple)
- **542,172 paragraphs**
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
F1 Macro:    0.533
F1 Weighted: 0.798
Precision:   0.648
Recall:      0.897
```

**Key finding**: Performance strongly correlates with sample size. Postures with 500+ samples achieve F1 > 0.92.

### Final Model Comparison (from `03_evaluation.ipynb`)
```
Metric          TF-IDF       LF (0.5)     LF (0.6)     LF Per-Class
F1 Micro        0.7524       0.6333       0.7018       0.7735
F1 Macro        0.5330       0.4718       0.5420       0.6000
Precision       0.6482       0.4785       0.5853       0.7116
Recall          0.8966       0.9363       0.8764       0.8472
```

**Best model**: Legal-Longformer with per-class threshold optimization
- **+2.1% F1 Micro** and **+6.7% F1 Macro** over TF-IDF baseline
- Per-class threshold range: 0.45–0.90, mean: 0.72
- Key insight: Default 0.5 threshold caused massive over-prediction (high recall, low precision). Per-class thresholds corrected this by finding the optimal cutoff per label on validation data.

### Threshold Optimization Impact
At default threshold (0.5), the transformer **underperformed** the baseline due to:
1. Over-prediction: recall 0.94 but precision only 0.48 (model predicted too many labels)
2. Limited compute: Apple MPS, batch_size=8, only 5 epochs
3. Summarization loss: 33% of docs lost information through LED compression

Per-class threshold optimization resolved the over-prediction issue, raising precision from 0.48→0.71 while keeping recall at 0.85. This turned a "failed" model into the **best performer**.

---

## Transformer Model: Legal-Longformer Hybrid

### Why Legal-Longformer (not Legal-BERT)?
- **Legal-BERT**: 512 token limit → covers only ~5% of documents fully
- **Legal-Longformer**: 4,096 token limit → covers ~65% of documents fully
- Model: `lexlms/legal-longformer-base` (same authors as Legal-BERT)

### Hybrid Strategy for Long Documents
Documents exceeding 4,096 tokens are **summarized** before classification:

| Document Length | Strategy | Model Used |
|-----------------|----------|------------|
| ≤4,096 tokens | Direct classification | Legal-Longformer |
| >4,096 tokens | Summarize → Classify | Legal-LED (16K) → Legal-Longformer |

This preserves semantic content better than arbitrary truncation.

### Implementation (`src/bert_trainer.py`)

```python
from src.bert_trainer import HybridLegalClassifier, TrainingConfig

# Initialize hybrid classifier
classifier = HybridLegalClassifier(
    num_labels=41,
    classifier_model='lexlms/legal-longformer-base',  # 4,096 tokens
    summarizer_model='nsi319/legal-led-base-16384',   # 16,384 tokens
    cache_dir='outputs/summaries',
)

# Train (automatically preprocesses long docs)
classifier.train(
    train_texts, y_train,
    val_texts, y_val,
    config=TrainingConfig(batch_size=4, num_epochs=5),
)

# Predict
y_pred = classifier.predict(test_texts)
```

### Key Classes
| Class | Purpose |
|-------|---------|
| `DeviceManager` | Auto-detect CUDA/MPS/CPU |
| `SummaryCache` | Disk cache for expensive summaries |
| `LegalSummarizer` | LED-based document summarization |
| `LegalLongformerTrainer` | Classification fine-tuning |
| `HybridLegalClassifier` | Orchestrates summarize + classify |
| `TrainingConfig` | Hyperparameter configuration |
| `TrainingHistory` | Training metrics tracking |

### Training Configuration (Actual)
```python
config = TrainingConfig(
    batch_size=8,
    gradient_accumulation_steps=4,  # Effective batch = 32
    learning_rate=1e-5,
    num_epochs=5,
    warmup_ratio=0.1,
    early_stopping_patience=2,
    use_pos_weight=True,  # Compensate for class imbalance
)
```

### Training History
- Best epoch: 4 (val_f1_micro=0.628)
- Training converged but val F1 plateaued around 0.62
- Early stopping did not trigger (5 epochs completed)

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
│   └── label_distribution.json   # Label statistics and frequencies
├── docs/
│   ├── legal_posture_ontology.md # Taxonomy with UML diagrams
│   ├── challenge_description.txt # OCR'd requirements
│   └── data_analysis.md
├── notebooks/
│   ├── 01_data_exploration.ipynb # Q1 - COMPLETE
│   ├── 02_modeling.ipynb         # Q2 - COMPLETE (baseline + transformer)
│   └── ontology_verification.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # DataLoader class
│   ├── data_analyzer.py    # DatasetAnalyzer, PostureTaxonomy
│   ├── model_trainer.py    # DataPreparer, BaselineTrainer (TF-IDF)
│   ├── model_evaluator.py  # MultiLabelEvaluator
│   ├── bert_trainer.py     # HybridLegalClassifier (Longformer)
│   └── visualization.py
└── outputs/
    ├── summaries/          # Cached document summaries
    ├── legal_longformer_best.pt  # Best model checkpoint
    └── *.history.json      # Training history
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

## Environment Setup

### Option 1: Conda (recommended)
```bash
# Create new environment
conda create -n tr_challenge python=3.11 -y
conda activate tr_challenge

# Install dependencies
pip install -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name=tr_challenge
```

### Option 2: venv
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=tr_challenge
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
```

### Data Setup
The data file `data/TRDataChallenge2023.txt` is **not in the repo** (per challenge rules).
Download from the challenge link and place in `data/` folder.

---

## Immediate TODO

1. ✅ **Run notebook**: Execute `02_modeling.ipynb` (baseline + transformer training)
2. ✅ **Run notebook**: Execute `03_evaluation.ipynb` (checkpoint evaluation + threshold optimization)
3. ✅ **Update CLAUDE.md**: Fill in actual results from notebook runs
4. ✅ **Create interview presentation** (PPT deck with speaker notes)
5. ⏳ **Write Question 3** response (next steps / production planning)
6. ⏳ **Package** for submission (zip without data)

## Results (All Notebooks Executed)

**Deliverables**: [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb), [02_modeling.ipynb](notebooks/02_modeling.ipynb), [03_evaluation.ipynb](notebooks/03_evaluation.ipynb)

### Final Model Comparison (from `03_evaluation.ipynb`)
```
Metric          TF-IDF       LF (0.5)     LF (0.6)     LF Per-Class
F1 Micro        0.7524       0.6333       0.7018       0.7735
F1 Macro        0.5330       0.4718       0.5420       0.6000
Precision       0.6482       0.4785       0.5853       0.7116
Recall          0.8966       0.9363       0.8764       0.8472
```

**Best model**: Legal-Longformer with per-class threshold optimization
- **+2.1% F1 Micro** and **+6.7% F1 Macro** over TF-IDF baseline
- Per-class threshold range: 0.45–0.90, mean: 0.72

### Baseline Top/Bottom Performers (TF-IDF)
```
TOP 3:    Appellate Review (F1=0.950), On Appeal (F1=0.927), Review of Admin Decision (F1=0.889)
BOTTOM 3: Motion to Set Aside (F1=0.192), Motion to Stay (F1=0.211), Motion for Costs (F1=0.216)
```

### Document Length Analysis
- Token mean: 3,786 | Token median: 2,693 | Token max: 72,080
- Only 4.1% of docs fit BERT's 512-token limit
- Longformer (4,096 tokens) covers 65.1% (11,194 docs direct classify)
- 33.1% (5,530 docs) required LED summarization before classification
- All 5,530 summaries cached to disk for reproducibility

### Automation Feasibility (TF-IDF Baseline)
```
Automatable (F1 >= 0.63):       13/41 postures
High confidence (F1 >= 0.93):    1/41 postures
Needs review (0.50-0.63):       11/41 postures
Not feasible (F1 < 0.50):       17/41 postures

Recommended tiered deployment:
  Full Automation:   1 posture  (Appellate Review, F1=0.95)
  Assisted Labeling: 12 postures (model + human review, F1 0.63-0.93)
  Human Review:      11 postures (model suggests, human decides)
  Human Only:        17 postures (insufficient model confidence)

~92% of labeling volume (by sample count) covered by automatable + review tiers.
LF Per-Class thresholds improve further (+2.1% F1 Micro, +6.7% F1 Macro).
```

### Threshold Optimization Impact
At default threshold (0.5), the transformer **underperformed** the baseline due to over-prediction (recall 0.94, precision 0.48). Root causes:
1. **Over-prediction**: Default 0.5 cutoff too low for imbalanced multi-label problem
2. **Limited compute**: Apple MPS (not GPU cluster), batch_size=8, only 5 epochs
3. **Summarization loss**: 33% of docs lost information through LED compression

Per-class threshold optimization (0.45–0.90 range, mean 0.72) resolved the over-prediction issue, raising precision from 0.48→0.71 while keeping recall at 0.85. This turned a "failed" model into the **best performer**.
