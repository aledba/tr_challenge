# Thomson Reuters Procedural Posture Classification
## Data Science Challenge - Exploratory Modeling Project

**Objective**: Build and evaluate models for automated classification of procedural postures in judicial opinions.

**Dataset**: 18,000 legal documents → 16,724 after filtering (41 viable posture labels with ≥50 samples, multi-label avg 1.54 labels/doc)

**Best Result**: F1 Micro 0.783 (Ensemble AND), with 18/41 classes exceeding human low agreement (κ=0.63)

---

## Quick Start

### Prerequisites
```bash
conda create -n tr_challenge python=3.11
conda activate tr_challenge
pip install -r requirements.txt
```

### Environment
- **Python**: 3.11
- **Key Libraries**: PyTorch 2.0, Transformers 4.36, scikit-learn 1.3, pandas, numpy
- **Hardware**: GPU/MPS recommended for notebook 02 training (~15 hours on M2)

### Running the Analysis
```bash
# 1. Data Exploration
jupyter notebook notebooks/01_data_exploration.ipynb

# 2. Baseline Model Training
jupyter notebook notebooks/02_modeling.ipynb

# 3. Model Evaluation (requires trained model from step 2)
jupyter notebook notebooks/03_evaluation.ipynb

# 4. Ensemble Analysis (requires predictions from steps 2 & 3)
jupyter notebook notebooks/04_ensemble_analysis.ipynb
```

### Project Structure
```
TR_hiring_exercise/
├── data/
│   ├── TRDataChallenge2023.txt           # JSON Lines format (not in repo)
│   └── label_distribution.json           # Label statistics and frequencies
├── docs/
│   ├── challenge_description.txt         # Original requirements
│   ├── data_analysis.md                  # Question 1 report
│   ├── legal_posture_ontology.md         # Domain analysis with UML
│   └── question_3_next_steps.md          # Improvement roadmap
├── notebooks/
│   ├── 01_data_exploration.ipynb         # EDA + statistics
│   ├── 02_modeling.ipynb                 # TF-IDF baseline + transformer training
│   ├── 03_evaluation.ipynb               # Threshold optimization + evaluation
│   └── 04_ensemble_analysis.ipynb        # Complementarity analysis
├── src/
│   ├── __init__.py
│   ├── data_loader.py                    # JSON Lines parser
│   ├── data_analyzer.py                  # Dataset analysis + taxonomy
│   ├── model_trainer.py                  # Data prep + TF-IDF
│   ├── bert_trainer.py                   # Transformer training
│   ├── model_evaluator.py                # Metrics + feasibility
│   └── visualization.py                  # Plotting utilities
├── tests/
│   ├── test_data_loader.py
│   ├── test_model_trainer.py
│   ├── test_model_evaluator.py
│   └── test_bert_trainer.py
├── outputs/
│   ├── legal_longformer_best.pt          # Trained model checkpoint
│   ├── tfidf_test_predictions.npz        # Baseline predictions
│   ├── legal_longformer_test_predictions.npz  # Transformer predictions
│   ├── *.history.json                    # Training history
│   └── summaries/                        # Cached document summaries (5,530 files)
└── requirements.txt
```

---

## Development Process

### Phase 1: Domain Analysis & Ontology (Pre-Modeling)

**Goal**: Understand legal domain structure before modeling.

**Approach**:
- Analyzed 224 posture labels using AI-assisted research and web searches
- Identified hierarchical relationships, mutual exclusions, contextual dependencies
- Documented findings in `legal_posture_ontology.md` with UML diagrams

**Key Findings**:
- **Multi-dimensional taxonomy**: Postures encode Stage (On Appeal), Motion Type (Motion to Dismiss), Proceeding Type (Family Law), etc.
- **Hierarchical structure**: Specific motions (e.g., "MTD for Lack of SMJ") may imply general category ("Motion to Dismiss")
- **Contextual dependencies**: Some motions only make sense in specific stages (e.g., appellate motions require appellate context)
- **Annotation policy unclear**: Dataset typically labels only leaf nodes, not parent categories

**Caveat**: This analysis lacks deep legal expertise and should be validated by domain experts.

**Outcome**: Identified potential for ontology-guided predictions as future improvement path.

---

### Phase 2: Data Exploration (Notebook 01)

**Goal**: Characterize dataset for modeling decisions.

**Key Statistics**:
- **Documents**: 18,000 raw → 16,724 after filtering (removed docs with only rare labels)
- **Postures**: 224 unique labels → filtered to 41 classes (≥50 samples each)
- **Paragraphs**: 542,172 total (avg 30 per doc)
- **Multi-label**: 49.8% of documents have multiple postures (avg 1.54 labels/doc)

**Class Distribution**:
- Top 3 classes: "On Appeal" (9,197), "Appellate Review" (4,652), "Review of Admin Decision" (2,773)
- Long tail: 183 classes with <50 samples (excluded from modeling)
- 41 viable classes cover 94.8% of all label occurrences

**Text Characteristics**:
- Median doc length: 2,693 tokens
- ~35% of docs exceed 4,096 tokens (Longformer limit)
- Longest: 72,080 tokens → requires summarization strategy

**Decision**: Use min_label_count=50 threshold to balance class coverage and model tractability.

**Deliverable**: `docs/data_analysis.md` (Question 1 report)

---

### Phase 3: Baseline Modeling (Notebook 02)

**Goal**: Establish fast, interpretable baseline.

**Model**: TF-IDF + Logistic Regression
- **Vectorizer**: max_features=10,000, ngram_range=(1,2), min_df=2
- **Classifier**: OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000))
- **Training time**: ~4 seconds on CPU

**Data Split**: 70/15/15 train/val/test (stratified by label frequency)
- Train: 11,706 docs
- Val: 2,509 docs
- Test: 2,509 docs

**Results** (Test Set):
- **F1 Micro**: 0.752
- **F1 Macro**: 0.533 (22pp gap indicates class imbalance)
- **Precision**: 0.648
- **Recall**: 0.897

**Strengths**:
- Fast inference (10ms/doc)
- Interpretable feature weights
- Strong on common, keyword-rich classes

**Weaknesses**:
- No semantic understanding
- Struggles with rare classes (17 classes F1 < 0.50)
- Limited context window (TF-IDF treats docs as bags of words)

**Feasibility Assessment**: Baseline exceeds human low agreement (κ=0.63) on 13/41 classes. However, these 13 classes cover the majority of labeling volume due to their high frequency.

---

### Phase 4: Advanced Modeling (Training + Notebook 03)

**Goal**: Improve performance with transformer-based model.

#### 4.1 Model Architecture

**Base Model**: `lexlms/legal-longformer-base`
- **Context window**: 4,096 tokens (vs. BERT's 512)
- **Architecture**: Longformer attention pattern (local + global)
- **Pre-trained on legal corpora** by the same authors as Legal-BERT

#### 4.2 Handling Long Documents

**Problem**: ~35% of documents exceed 4,096 token limit (up to 72K tokens)

**Solution**: Hybrid approach with automatic summarization
- **Direct classification**: Docs ≤4,096 tokens → use full text
- **Summarization**: Docs >4,096 tokens → summarize to fit context, then classify
- **Summarization model**: `nsi319/legal-led-base-16384` (Legal-LED, 16K context)
- **Caching**: Summaries saved to `outputs/summaries/` (one-time cost)

**Impact**: 5,530 of 16,724 docs required summarization (33%)

#### 4.3 Training Configuration

**Hyperparameters**:
- **Optimizer**: AdamW (lr=1e-5, weight_decay=0.01)
- **Epochs**: 5 (early stopping on validation F1)
- **Batch size**: 8 (effective 32 with gradient accumulation)
- **Loss**: Binary cross-entropy with logits
- **Warmup**: 10% of training steps
- **Hardware**: Apple M2 MPS (192GB RAM), ~15 hours training time

**Threshold Optimization**:
- Initial: Global threshold = 0.5
- Validation sweep: Tested thresholds 0.1 to 0.8
- Optimal: Per-class thresholds (range 0.45–0.90, mean 0.72)

#### 4.4 Results (Test Set)

| Configuration | F1 Micro | F1 Macro | Precision | Recall |
|---------------|----------|----------|-----------|--------|
| Threshold=0.5 | 0.633 | 0.472 | 0.479 | 0.936 |
| Threshold=0.6 | 0.702 | 0.542 | 0.585 | 0.876 |
| **Per-class thresholds** | **0.774** | **0.600** | **0.712** | **0.847** |

**Improvement over TF-IDF**: +2.1pp F1 Micro, +6.7pp F1 Macro with per-class thresholds

**Key Insight**: At default threshold 0.5, the transformer appeared to fail (F1 Micro 0.633 vs baseline 0.752). The problem was over-prediction: recall 0.94 but precision only 0.48. Per-class threshold optimization (finding optimal cutoff per label on validation data) fixed this, turning a "failed" model into the best single model.

**Strengths**:
- Semantic understanding of legal context
- Better on rare classes (+6.7pp F1 Macro)
- Higher precision with per-class thresholds (0.712 vs. 0.648)

**Weaknesses**:
- 50ms inference latency per doc (5x slower than TF-IDF)
- Still struggles on rare classes with limited training data
- Summarization introduces potential information loss for 33% of docs

---

### Phase 5: Ensemble Analysis (Notebook 04)

**Goal**: Quantify model complementarity and evaluate ensemble strategies.

#### 5.1 Complementarity Analysis

**Prediction Agreement**: 96.6% overall
- Both predict positive: 4.6%
- Both predict negative: 91.9%
- Only TF-IDF positive: 0.6%
- Only Longformer positive: 2.8%

**Per-Class Strengths** (non-overlapping sets):
- **TF-IDF excels**: 21 classes (F1 gap ≥+0.05 over Longformer)
- **Longformer excels**: 5 classes (F1 gap ≥+0.05 over TF-IDF)
- **Similar performance**: 15 classes (within ±0.05 F1)

**Key Insight**: High agreement but clear complementarity. TF-IDF dominates on keyword-rich classes; Longformer adds value on semantically complex ones.

#### 5.2 Ensemble Strategies Tested

| Strategy | Logic | F1 Micro | Improvement |
|----------|-------|----------|-------------|
| TF-IDF alone | Baseline | 0.7524 | — |
| Longformer alone | Threshold=0.5 | 0.6333 | -11.9pp |
| **Intersection (AND)** | Both predict 1 | **0.7827** | **+3.03pp** ✓ |
| Union (OR) | Either predicts 1 | 0.6181 | -13.4pp |
| Weighted | TF-IDF on strong classes | 0.7403 | -1.2pp |

**Winner**: Intersection (AND) strategy
- **Rationale**: When both models agree on a positive prediction, it's very likely correct
- **Tradeoff**: Boosts precision to 0.713, maintains recall at 0.868
- **Implementation**: `y_pred_ensemble = np.minimum(y_pred_tfidf, y_pred_longformer)`

#### 5.3 Classes Still Struggling

**13 classes** perform F1 < 0.50 on BOTH models:
- Indicates insufficient training data, not model architecture issue
- Examples: "Motion for Permanent Injunction" (F1=0.16), "Motion for Costs" (F1=0.20)
- Average support: 15 samples (vs. 180 for viable classes)

**Implication**: Ensemble won't fix data scarcity; need targeted data augmentation.

---

## Results Summary

### Model Performance Comparison (Test Set)

| Model | F1 Micro | F1 Macro | Precision | Recall | Classes F1≥0.63 |
|-------|----------|----------|-----------|--------|-----------------|
| TF-IDF Baseline | 0.752 | 0.533 | 0.648 | 0.897 | 13/41 |
| LF Per-Class Thresh. | 0.774 | 0.600 | 0.712 | 0.847 | 23/41 |
| **Ensemble (AND)** | **0.783** | **0.575** | **0.713** | **0.868** | **18/41** |

**Note**: The ensemble uses LF predictions at default threshold (0.5), not per-class optimized. The intersection strategy naturally filters out the LF over-predictions, achieving the highest F1 Micro overall.

### Feasibility Recommendation

**Verdict**: Automation is **feasible** for procedural posture classification with human-in-the-loop.

**Rationale**:
- Ensemble achieves F1 Micro 0.783, comparable to human moderate agreement
- 18/41 classes (44%) exceed minimum automation threshold (F1≥0.63)
- These high-performing classes cover the majority of labeling volume (~92%) due to their frequency

**Deployment Strategy**:
- **Full automation**: 1 class with F1≥0.93 (Appellate Review, highest confidence)
- **Assisted automation**: 12–17 classes with 0.63≤F1<0.93 (model suggests, human reviews)
- **Human review**: ~11 classes with 0.50≤F1<0.63 (model assists, human decides)
- **Manual only**: ~17 classes with F1<0.50 (insufficient model confidence)

**ROI Estimate**: Automating the top-performing classes covers ~92% of label volume, significantly reducing annotation time while maintaining quality through human oversight on uncertain predictions.

---

## Technical Details

### Model Artifacts

**Saved Files**:
- `outputs/legal_longformer_best.pt` - Model checkpoint (epoch 4, 1.2 GB)
- `outputs/legal_longformer_best.history.json` - Training curves
- `outputs/tfidf_test_predictions.npz` - TF-IDF predictions (y_pred, y_proba)
- `outputs/legal_longformer_test_predictions.npz` - Longformer predictions (y_pred, y_proba)
- `outputs/summaries/*.json` - Cached document summaries (5,530 files)

**NPZ Format**: NumPy compressed archive containing:
```python
data = np.load('predictions.npz')
y_pred = data['y_pred']    # shape: (2509, 41), binary predictions
y_proba = data['y_proba']  # shape: (2509, 41), probability scores
```

### Compute Requirements

**Training**:
- Legal-Longformer: ~15 hours on Apple M2 MPS (192GB RAM)
- TF-IDF: <1 minute on CPU

**Inference** (batch_size=16):
- TF-IDF: 10 ms/doc
- Legal-Longformer: 50 ms/doc (includes preprocessing)
- Ensemble: 60 ms/doc (parallel prediction + voting)

**Storage**:
- Models: 1.2 GB (Legal-Longformer) + 50 MB (TF-IDF)
- Summaries cache: 150 MB (5,530 files)
- Predictions: 2 MB per model per test set

### Label Statistics

**Class Distribution** (41 classes, ≥50 samples):
- **Automatable** (F1≥0.63): 13 classes (TF-IDF baseline), up to 23 with per-class thresholds
- **Needs review** (0.50≤F1<0.63): 11 classes
- **Not feasible** (F1<0.50): 17 classes

**Annotation Agreement** (from challenge description):
- Human κ high: 0.93 (e.g., "Appellate Review")
- Human κ low: 0.63 (e.g., "Trial or Guilt Phase Motion")
- Model target: Match or exceed κ=0.63 for automation feasibility

### Data Preprocessing

**Text Extraction**:
- Concatenate all paragraphs within sections
- Preserve section boundaries with special tokens (not implemented in baseline)
- Remove empty paragraphs

**Long Document Handling**:
- Tokenize with `lexlms/legal-longformer-base` tokenizer
- If ≤4,096 tokens: Use full text
- If >4,096 tokens: Summarize using Legal-LED
  - Target length: ~1,024 tokens (leaves buffer for classification)
  - Summarization params: max_length=1024, min_length=256, max_input=16384
  - Cache location: `outputs/summaries/{doc_id}.json`

**Label Encoding**:
- Multi-hot encoding: 41-dimensional binary vector per document
- Positive class weight: sqrt(neg_samples / pos_samples) for imbalanced classes

---

## Next Steps

See `docs/question_3_next_steps.md` for detailed improvement roadmap.

**Summary**: Three complementary paths identified:

1. **Ontology-Guided Predictions** (explore hierarchical constraints)
   - Post-processing layer to enforce legal taxonomy consistency
   - Potential gain: +1-3pp F1 Micro from refined predictions

2. **Ensemble Optimization** (✓ already validated +3pp gain)
   - Per-class learned weights instead of simple intersection
   - Stacking meta-learner with document metadata features
   - Potential gain: +4-5pp F1 Micro total

3. **Targeted Data Augmentation** (address 13 struggling classes)
   - LLM-based paraphrasing for rare classes
   - Generate 3-5 synthetic examples per original document
   - Potential gain: +2-3pp F1 Micro overall, +10-15pp on rare classes

**Recommended Priority**: Start with ensemble optimization (Path 2) - low risk, validated benefit, no retraining required.

---

## Dependencies

See `requirements.txt` for complete list. Key packages:

```
torch==2.0.1
transformers==4.36.0
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.26.2
matplotlib==3.8.2
seaborn==0.13.0
jupyter==1.0.0
```

**Note**: For reproducibility, install exact versions. Training may require adjustment of batch sizes based on available GPU memory.

---

## Citations & Acknowledgments

**Pre-trained Models**:
- Legal-Longformer: `lexlms/legal-longformer-base` - Longformer pre-trained on legal corpora
- Legal-LED: `nsi319/legal-led-base-16384` - Legal domain Longformer Encoder-Decoder
- Longformer: Beltagy et al. (2020) - Efficient attention for long documents

**Code References**:
- Hugging Face Transformers library for model implementation
- Scikit-learn for baseline models and metrics
- Standard PyTorch training loops (adapted from official examples)

**Domain Research**:
- AI-assisted analysis (Claude) for legal taxonomy exploration
- Web research on procedural posture definitions and hierarchies

---

## Contact

This project was completed as part of the Thomson Reuters Data Science Challenge (January 2026).

For questions about methodology or implementation details, refer to inline code comments and notebook markdown cells.
