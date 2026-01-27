# SOTA: Procedural Posture Classification

## The Benchmark Dataset

**POSTURE50K** (Song et al., 2022) is the standard benchmark for this exact task:

- 50,000 US legal opinions with procedural posture labels
- Multi-label classification with Zipfian label distribution
- Inter-annotator agreement: **κ = 0.81** (top 10 labels)
- Covers all 50 US states, 2013–2020

Highly comparable to TR challenge data (18K docs, 224 postures, κ = 0.63–0.93).

---

## SOTA Performance

### LAMT_MLC (Song et al., Information Systems 2022)

Best published system for POSTURE50K:

| Component | Details |
|-----------|---------|
| Base model | RoBERTa-Large + continued legal pre-training |
| Key technique | Label-attention mechanism for class imbalance |
| Multi-task | Secondary objective for low-frequency classes |

### LexGLUE Benchmark (Legal Document Classification)

| Model | μ-F1 | m-F1 |
|-------|------|------|
| Legal-BERT | **79.8** | **72.0** |
| RoBERTa-Large | 79.4 | 70.8 |
| Longformer | 78.5 | 70.5 |
| BERT-base | 77.8 | 69.5 |

### Our Baseline Comparison

| Model | μ-F1 | m-F1 |
|-------|------|------|
| TF-IDF + LogReg (ours) | 75.2 | 52.6 |
| SOTA transformers | ~80 | ~72 |
| **Gap** | +5pp | +20pp |

Takeaway: Transformers offer modest μ-F1 gains but significant m-F1 improvement on rare classes.

---

## Commercial Systems

### Thomson Reuters Westlaw Precision

TR already does procedural posture tagging in production:

- Procedural Posture Filters for search refinement
- Tags documents by "law, facts, outcomes, procedural posture, parties"
- **250 attorneys hired** for human annotation + ML hybrid
- Released 2022

### Other Players

| Company | Product | Notes |
|---------|---------|-------|
| LexisNexis | Lexis+ AI | RAG with Claude 3, GraphRAG |
| Harvey AI | Vault | GPT-4 fine-tuned on legal data |
| Casetext | CoCounsel | Acquired by TR for $650M (2023) |

---

## Key Technical Insights

### Document Length

| Model | Max Tokens | Doc Coverage |
|-------|------------|--------------|
| BERT / Legal-BERT | 512 | ~5% |
| Longformer | 4,096 | ~65% |
| Legal-Longformer 8192 | 8,192 | ~85% |
| Legal-LED | 16,384 | ~95% |

Our hybrid approach (summarize long docs → classify) aligns with current research.

### Label Imbalance

LAMT_MLC's label-attention mechanism directly addresses the macro-F1 problem:
- Uses label embeddings to bridge semantic gap between docs and rare classes
- Multi-task learning with secondary focus on low-frequency postures

### Zero-Shot Evaluation

POSTURE50K includes splits for zero-shot evaluation on rare classes—relevant for our 140 postures with <10 samples.

---

## References

1. Song et al. (2022). "Multi-label legal document classification: A deep learning-based approach with label-attention and domain-specific pre-training." *Information Systems*, 106. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0306437921000016)

2. Chalkidis et al. (2022). "LexGLUE: A Benchmark Dataset for Legal Language Understanding in English." *ACL 2022*. [GitHub](https://github.com/coastalcph/lex-glue)

3. Mamakas et al. (2022). "Processing Long Legal Documents with Pre-trained Transformers." *NLLP Workshop*. [ACL Anthology](https://aclanthology.org/2022.nllp-1.11.pdf)

4. Thomson Reuters (2022). "Westlaw Precision" announcement. [LawNext](https://www.lawnext.com/2022/09/thomson-reuters-unveils-next-generation-of-westlaw-aiming-to-make-legal-research-results-more-precise.html)

5. Xenouleas et al. (2024). "Natural Language Processing for the Legal Domain: A Survey." [arXiv](https://arxiv.org/html/2410.21306v3)
