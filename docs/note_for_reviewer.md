# Note for the Reviewer

**Thomson Reuters Data Science Challenge**
*Procedural Posture Classification*

---

## My Approach

### 1. Business Understanding

I started by deeply understanding what procedural postures are in the legal domain. This led me to create a formal **ontology** representing the 224 labels as a UML class diagram ([docs/legal_posture_ontology.md](legal_posture_ontology.md)).

This exercise had two goals:
1. **Domain comprehension** — understanding relationships between postures (IS-A hierarchies, dependencies, orthogonal dimensions)
2. **Data insights** — discovering that labels encode multiple facets (Stage, Motion Type, Proceeding) that combine compositionally, not hierarchically

Key finding: labels like "Motion to Post Bond" *depend on* "On Appeal" (contextual requirement), they don't inherit from it.

### 2. Data Exploration

In [notebooks/01_data_exploration.ipynb](../notebooks/01_data_exploration.ipynb), I analyzed the dataset structure:
- 18,000 documents with nested sections/paragraphs
- 224 postures with severe class imbalance (top label: 26% of docs; 140 labels have <10 samples)
- Multi-label: 49.8% of documents have multiple postures

### 3. Modeling

In [notebooks/02_modeling.ipynb](../notebooks/02_modeling.ipynb):

**TF-IDF Baseline**: Logistic Regression with TF-IDF features achieved F1 Micro: 0.751.

**Transformer Approach**: I chose `lexlms/legal-longformer-base` (4,096 token context) over Legal-BERT (512 tokens) because document length analysis showed only ~5% of documents fit within BERT's window vs ~67% for Longformer.

**Long Document Strategy**: For documents exceeding 4,096 tokens (~33%), I used `nsi319/legal-led-base-16384` to generate semantic summaries before classification. Summaries are cached in `outputs/summaries/` for reproducibility.

The training journey is documented in [docs/training_story.md](training_story.md). Key learnings:
- `pos_weight` in BCELoss is critical — without it, F1 collapsed to 0.02
- Learning rate "Goldilocks zone": 2e-5 (catastrophic forgetting) → 5e-6 (too conservative) → **1e-5** (optimal)
- Per-class threshold optimization provided significant gains

### 4. Evaluation

In [notebooks/03_evaluation.ipynb](../notebooks/03_evaluation.ipynb), I applied proper methodology:
- Thresholds optimized on **validation set**
- Final evaluation on **held-out test set**

| Model | F1 Micro | F1 Macro |
|-------|----------|----------|
| TF-IDF Baseline | 0.751 | 0.521 |
| Legal-Longformer (per-class thresh) | 0.770 | 0.585 |
| Ensemble (AND) | **0.783** | 0.575 |

### 5. Next Steps

Documented in [docs/question_3_next_steps.md](question_3_next_steps.md), I proposed three improvement paths:
1. **Ontology-guided predictions** — exploit label dependencies
2. **Ensemble optimization** — weighted voting based on per-class strengths
3. **Data augmentation** — address rare class performance

---

## Use of AI Assistance

I used **Claude** (Anthropic) for:
- Business analysis and legal domain research
- Generation of boilerplate code and initial implementations

**My personal contributions**:
- Verified all code correctness through testing and 44 unit tests
- Steered the solution architecture (OOP patterns, modular design)
- Made all data science decisions (model selection, hyperparameter tuning, evaluation methodology)
- Ensured proper train/val/test separation to prevent data leakage
- Created the ontology analysis and domain modeling

The final codebase reflects my standards: logic in `src/` modules, notebooks as consumers, comprehensive error handling, and reproducible results.

---

*January 2026*
