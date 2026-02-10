# Question 3: Next Steps for Procedural Posture Classification
## Strategic Roadmap for Model Improvement

**Prepared for:** Thomson Reuters Data Science Challenge  
**Date:** January 28, 2026  
**Focus:** Model improvement pathway

---

## Executive Summary

I am choosing to focus on **model improvement** rather than production deployment. Model performance needs enhancement before investing in production infrastructure, and these improvements will be necessary for production readiness anyway.

This document outlines three complementary improvement paths for the procedural posture classification system (currently F1 Micro: 0.783):

1. **Ontology-Guided Predictions** - Leverage legal domain structure
2. **Ensemble Optimization** - Combine model strengths  
3. **Targeted Data Augmentation** - Address rare class performance

---

## Current State Assessment

### Model Performance (Test Set)

| Model | F1 Micro | F1 Macro | Precision | Recall | Notes |
|-------|----------|----------|-----------|--------|-------|
| TF-IDF Baseline | 0.752 | 0.533 | 0.744 | 0.760 | Fast, interpretable |
| Legal-Longformer | 0.770 | 0.560 | 0.794 | 0.748 | With per-class thresholds |
| Ensemble (AND) | **0.783** | **0.575** | **0.801** | **0.766** | Current best |

### Key Strengths
- **Solid baseline performance**: Exceeds human low agreement (κ=0.63) on 28/41 classes
- **Model complementarity**: TF-IDF excels on 21 classes, Legal-Longformer on 5 classes
- **High precision**: 0.80+ across all configurations, suitable for automation assistance

### Key Challenges
- **Rare class performance**: 13 classes below F1=0.50 on both models
- **Macro-Micro gap**: 21pp difference indicates class imbalance issues
- **Threshold sensitivity**: Significant performance variation (0.633 → 0.770) with optimization

---

## Improvement Path 1: Ontology-Guided Predictions

### Rationale

**Important caveat**: This analysis was conducted without deep legal domain expertise. Using AI-assisted research and web searches, I explored the structure of the 224 posture labels to identify potential patterns (documented in `legal_posture_ontology.md`). A legal expert should validate these findings.

That said, the exploratory analysis suggests clear structural constraints in the data. For example:
- **Contextual dependencies**: "Motion to Post Bond" and "Motion to Expand the Record" appear to be appellate-specific motions that logically require an appellate stage context like "On Appeal" or "Appellate Review"
- **Mutual exclusions**: Motions with opposing outcomes (e.g., "Motion to Dismiss - Granted" vs. "Motion to Dismiss - Denied") cannot co-occur
- **Hierarchical relationships**: Specific dismissal types (e.g., "Motion to Dismiss for Lack of Subject Matter Jurisdiction") may imply the general "Motion to Dismiss" category

Current models ignore this domain structure entirely, leading to logically inconsistent predictions (e.g., predicting an appellate-specific motion without an appellate stage). An important observation: when hierarchies exist in the dataset, typically only the most specific (child) label is tagged, not both parent and child. For example, a specific appeal motion may not also be tagged with "On Appeal." The annotation policy here is unclear - whether this is intentional (only tag the leaf node) or represents missing labels. I worked with the data as-provided without attempting to modify the labeling.

### Technical Approach (if validated by business)

**Constraint Integration**
- Implement mutual exclusion rules (e.g., "Granted" ↔ "Denied")
- Add hierarchical consistency (child implies parent presence)
- Post-processing layer: adjust predictions to satisfy constraints
- This can be done without retraining models

**Hierarchical Loss Function**  
- Modify training objective to penalize constraint violations
- Weight losses by taxonomy depth (specificity)
- Compare against post-processing approach to validate benefit

**Hierarchy-Aware Embeddings**
- Encode parent-child relationships in label space
- Use hierarchical label smoothing during training
- More complex but may yield stronger improvements

### Expected Outcomes
- **Prediction consistency**: 100% constraint satisfaction (vs. current ~92%)
- **Performance gain**: +1-3pp F1 Micro from refined predictions
- **Interpretability**: Predictions align with legal taxonomy structure
- **Business value**: Reduced need for manual review of contradictory predictions

### Key Requirements
- Domain expertise to validate constraint rules
- Access to legal taxonomy documentation
- Minimal computational resources (post-processing approach)
- More significant compute if retraining with hierarchical loss

### Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Taxonomy constraints too rigid | May reduce recall | Make constraints soft (penalty vs. hard constraint) |
| Domain expertise bottleneck | Delays in validation | Document rules early, automate validation |
| Overfitting to structure | Poor generalization | Validate on diverse document types |

---

## Improvement Path 2: Ensemble Optimization

### Rationale
Empirical analysis shows clear model complementarity: TF-IDF excels on 21 classes, Legal-Longformer on 5 classes (non-overlapping sets), with the remaining 15 classes showing similar performance. Despite 96.6% prediction agreement overall, simple intersection voting already yields +3.03pp improvement.

### Technical Approach

**Per-Class Ensemble Weights**
- Use validation set to learn optimal weights per class
- Weighted probability averaging: `p_final = α * p_tfidf + (1-α) * p_longformer`
- Dynamic threshold selection per class
- Can implement quickly with existing predictions

**Stacking Meta-Learner**
- Train lightweight meta-classifier (logistic regression, XGBoost)
- Features: both model probabilities + document metadata (length, court, date)
- Cross-validation to prevent overfitting
- More sophisticated but requires careful validation

**Model Diversification**
- Add third model: RoBERTa-Legal or DeBERTa-v3
- Analyze new model's complementarity patterns
- Re-optimize ensemble weights with 3+ models
- Increases inference cost but may improve performance

### Expected Outcomes
- **Immediate gain**: +3pp from intersection strategy (already validated)
- **Optimized gain**: +4-5pp F1 Micro with learned weights
- **Stacking gain**: Additional +1-2pp with meta-learner
- **Total potential**: 5-6pp over TF-IDF baseline (→ 0.80-0.81 F1 Micro)
- **Low risk**: Working with existing predictions, no retraining required initially

### Key Requirements
- Standard ML infrastructure for meta-learner training
- Careful validation to avoid overfitting
- Use existing train/val/test splits
- Potential increased inference latency (multiple models)

### Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Overfitting to validation set | Poor test performance | Nested cross-validation, hold-out final test set |
| Increased inference latency | Deployment complexity | Parallelize model inference, optimize serving |
| Diminishing returns | Wasted effort | Monitor validation curves, stop if gains < 0.5pp |

---

## Improvement Path 3: Targeted Data Augmentation

### Rationale
13 classes perform below F1=0.50 on both models. Manual annotation is expensive and slow, but these classes clearly lack sufficient training data. Modern paraphrasing techniques (GPT-4, Claude) offer a cost-effective alternative to generate synthetic training examples.

### Technical Approach

**Focus on Paraphrasing**
- Use LLMs to rewrite case summaries from rare classes
- Preserve legal meaning while varying sentence structure and vocabulary
- Generate 3-5 paraphrases per original document for struggling classes
- Sample manual review (10-20%) to ensure quality doesn't degrade

**Implementation**
- Start with the 5 worst-performing classes as pilot
- Validate that synthetic data improves validation F1 before scaling
- If successful, expand to all 13 struggling classes
- Retrain models with augmented dataset

### Expected Outcomes
- **Rare class boost**: +10-15pp F1 on targeted classes
- **Overall improvement**: +2-3pp F1 Micro
- **Cost-effective**: Synthetic generation vs. manual annotation

### Key Requirements
- LLM API access for paraphrasing
- Domain expert for quality spot-checks
- Computational resources for retraining
- Risk: Synthetic artifacts could hurt generalization

### Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Synthetic data introduces artifacts | Model learns spurious patterns | Quality control, diversity metrics, held-out validation |
| Annotation budget overrun | Reduced sample size | Start with highest-impact classes, negotiate batch pricing |
| External data mismatch | Negative transfer | Careful domain analysis, ablation studies |
| Diminishing returns on rare classes | Wasted effort | Set minimum class size (50→100 samples), accept some classes as hard |

---

## Recommended Approach

### Sequencing Strategy

The three improvement paths can be pursued with different levels of effort and risk:

**Start with Ensemble Optimization** (Path 2)
- **Rationale**: +3pp improvement already validated, works with existing predictions
- **Advantage**: No retraining, fast to implement, low risk
- **First step**: Implement per-class weights and test on validation set
- **Next step**: Evaluate stacking if simple weighting plateaus

**Add Ontology Constraints in Parallel** (Path 1)
- **Rationale**: Constraint post-processing requires domain expertise input
- **Advantage**: Can be developed independently while ensemble work proceeds
- **First step**: Document mutual exclusion and hierarchical rules
- **Next step**: Implement post-processing layer and measure consistency improvements

**Consider Data Augmentation if Needed** (Path 3)
- **Rationale**: Most resource-intensive, addresses specific struggling classes
- **Advantage**: Directly tackles the 13 classes where both models fail
- **First step**: Analyze error patterns on low-performing classes
- **Next step**: Pilot synthetic data generation on 2-3 classes before scaling

### Success Indicators

Progress can be measured through:
- **F1 Micro**: Track improvements from baseline 0.752 toward 0.80+ target
- **Per-class coverage**: Monitor how many classes exceed human low agreement (F1 ≥ 0.63)
- **Prediction consistency**: Measure constraint violation rates
- **Inference latency**: Ensure ensemble doesn't degrade user experience

---