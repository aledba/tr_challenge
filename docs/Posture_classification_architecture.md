# Procedural Posture Classification — System Architecture

## 1. Overview

This document describes the software architecture of the **Procedural Posture Classification** system: a multi-label text classification pipeline for legal judicial opinions. The system classifies documents into 41 procedural posture labels using two complementary models (TF-IDF baseline and Legal-Longformer transformer) with ensemble capabilities.

### 1.1 Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Separation of Concerns** | 6 dedicated modules: loader, analyzer, trainer, evaluator, bert_trainer, visualization |
| **Dependency Injection** | `DatasetAnalyzer` and `DataPreparer` receive `DataLoader` rather than creating it |
| **Strategy Pattern** | `LongDocumentStrategy` ABC with Truncate / HeadTail / Summarize implementations |
| **Template Method** | `BaseTransformerTrainer.train()` defines the training loop; `_create_model()` is abstract |
| **Facade** | `HybridLegalClassifier` orchestrates `LegalSummarizer` + `LegalLongformerTrainer` |
| **Lazy Loading** | Models and tokenizers loaded on first use (`DataLoader._df`, `LegalSummarizer._model`) |
| **Disk Caching** | `SummaryCache` persists expensive LED summaries to `outputs/summaries/` |

### 1.2 UML Notation Key

| Relationship | Arrow | Mermaid | Meaning |
|---|---|---|---|
| Dependency | dashed, open arrow | `..>` | "uses" |
| Association | solid, open arrow | `-->` | "has reference to" |
| Inheritance | solid, closed arrow | `<\|--` | "extends" |
| Realization | dashed, closed arrow | `<\|..` | "implements abstract" |
| Composition | solid, filled diamond | `*--` | "owns lifecycle" |
| Aggregation | solid, open diamond | `o--` | "has, independent lifecycle" |

---

## 2. Domain Model

The input data consists of legal judicial opinions with nested text and multi-label posture annotations.

```mermaid
classDiagram
    class Document {
        +str documentId
        +List~str~ postures
        +List~Section~ sections
    }

    class Section {
        +List~Paragraph~ paragraphs
    }

    class Paragraph {
        +str text
    }

    class PostureLabel {
        <<enumeration>>
        On Appeal
        Appellate Review
        Motion to Dismiss
        ...41 viable labels
    }

    class PostureDimension {
        <<enumeration>>
        Stage
        MotionType
        CaseCharacteristic
        ProceduralEvent
        Proceeding
    }

    Document *-- "1..*" Section : contains
    Section *-- "1..*" Paragraph : contains
    Document --> "1..*" PostureLabel : annotated with
    PostureLabel --> PostureDimension : belongs to

    note for Document "18,000 documents\n49.8% multi-label\navg 1.54 labels/doc"
    note for PostureLabel "224 unique → filtered to 41\n(min 50 samples each)"
```

### 2.1 Posture Label Relationships

```mermaid
classDiagram
    class Stage {
        On Appeal
        Appellate Review
        Review of Admin Decision
    }

    class MotionType {
        Motion to Dismiss
        Motion for Summary Judgment
        Motion for Attorney Fees
    }

    class SpecificMotion {
        MTD for Lack of SMJ
        MTD for Lack of Personal J.
        MTD for Lack of Standing
    }

    class Proceeding {
        Family Law
        Criminal
        Bankruptcy
    }

    MotionType <|-- SpecificMotion : IS-A
    SpecificMotion ..> Stage : DEPENDS-ON
    Proceeding ..> Stage : co-occurs with

    note for SpecificMotion "Sparse labeling: only\nmost-specific label applied\n(3-13% co-occurrence with parent)"
    note for Stage "Dense labeling: stage\nalways co-labeled\n(89-100% co-occurrence)"
```

---

## 3. Package / Component Diagram

### 3.1 Data & Analysis Modules

```mermaid
flowchart TB
    subgraph DL_mod["📂 data_loader.py"]
        direction TB
        DL_1([DataLoader])
    end

    subgraph DA_mod["📂 data_analyzer.py"]
        direction TB
        DA_1([DatasetAnalyzer])
        DA_2([DatasetStatistics])
        DA_3([PostureTaxonomy])
    end

    DA_mod -.->|"aggregation"| DL_mod
    DL_mod -->|uses| PD([pandas])
    DA_mod -->|uses| PD

    style DL_mod fill:#2E5090,color:#fff
    style DA_mod fill:#2E5090,color:#fff
```

### 3.2 Baseline Training Module

```mermaid
flowchart TB
    subgraph MT_mod["📂 model_trainer.py"]
        direction TB
        MT_1([TextExtractor])
        MT_2([DataPreparer])
        MT_3([PreparedData])
        MT_4([BaselineTrainer])
    end

    MT_mod -.->|"aggregation"| DL(["📂 data_loader.DataLoader"])
    MT_mod -->|uses| SK([scikit-learn])
    MT_mod -->|uses| PD([pandas])

    style MT_mod fill:#457B9D,color:#fff
```

### 3.3 Transformer Training Module

```mermaid
flowchart TB
    subgraph BT_mod["📂 bert_trainer.py"]
        direction TB
        subgraph Utilities
            BT_1([DeviceManager])
            BT_2([TrainingConfig])
            BT_3([TrainingHistory])
        end
        subgraph Summarization
            BT_4([SummaryCache])
            BT_5([LegalSummarizer])
        end
        subgraph Strategy["Long Document Strategies"]
            BT_7([LongDocumentStrategy])
            BT_8([TruncateStrategy])
            BT_9([HeadTailStrategy])
            BT_10([SummarizeStrategy])
        end
        subgraph Training
            BT_6([LegalTextDataset])
            BT_11([BaseTransformerTrainer])
            BT_12([LegalLongformerTrainer])
        end
        subgraph Facade
            BT_13([HybridLegalClassifier])
        end
    end

    BT_mod -->|uses| PT([PyTorch])
    BT_mod -->|uses| HF([transformers])

    style BT_mod fill:#E85D04,color:#fff
```

### 3.4 Evaluation Module

```mermaid
flowchart TB
    subgraph ME_mod["📂 model_evaluator.py"]
        direction TB
        ME_1([MultiLabelEvaluator])
        ME_2([EvaluationResults])
        ME_3([ModelComparison])
        ME_4([compute_threshold_analysis])
        ME_5([save / load_predictions])
        ME_6([create_classification_report_df])
    end

    ME_mod -->|uses| SK([scikit-learn])

    style ME_mod fill:#457B9D,color:#fff
```

### 3.5 Visualization Module

```mermaid
flowchart TB
    subgraph VZ_mod["📂 visualization.py"]
        direction TB
        VZ_1([COLORS])
        VZ_2([setup_style])
        VZ_3([_add_chart_branding])
        VZ_4([plot_posture_distribution])
        VZ_5([plot_text_length_distribution])
        VZ_6([plot_class_imbalance])
    end

    VZ_mod -.->|"dependency"| DA(["📂 data_analyzer.DatasetStatistics"])
    VZ_mod -->|uses| MP([matplotlib / seaborn])

    style VZ_mod fill:#3D8B6F,color:#fff
```

### 3.6 Module Dependency Overview

```mermaid
flowchart LR
    DL["data_loader.py"]
    DA["data_analyzer.py"]
    MT["model_trainer.py"]
    ME["model_evaluator.py"]
    BT["bert_trainer.py"]
    VZ["visualization.py"]

    DA -.->|aggregation| DL
    MT -.->|aggregation| DL
    VZ -.->|dependency| DA

    style DL fill:#2E5090,color:#fff
    style DA fill:#2E5090,color:#fff
    style MT fill:#457B9D,color:#fff
    style ME fill:#457B9D,color:#fff
    style BT fill:#E85D04,color:#fff
    style VZ fill:#3D8B6F,color:#fff
```

---

## 4. Class Diagrams

### 4.1 Data Layer (`data_loader.py` + `data_analyzer.py`)

```mermaid
classDiagram
    class DataLoader {
        +Path file_path
        -Optional~DataFrame~ _df
        +__init__(file_path: str | Path)
        +load() DataFrame
        +peek(n: int) DataFrame
        +get_schema() dict
        +get_column_names() list~str~
        +df DataFrame
    }

    class DatasetAnalyzer {
        +DataLoader loader
        -Optional~DatasetStatistics~ _stats
        -Optional~str~ _posture_col
        -Optional~str~ _paragraphs_col
        -Optional~str~ _text_col
        +__init__(loader: DataLoader)
        -_detect_columns(df: DataFrame) None
        +compute_statistics(use_cache: bool) DatasetStatistics
        +get_posture_distribution() Series
        +get_class_imbalance_ratio() float
        +get_word_counts() Series
        +get_posture_taxonomy() PostureTaxonomy
    }

    class DatasetStatistics {
        <<dataclass>>
        +int num_documents
        +int num_postures
        +int total_paragraphs
        +Series posture_distribution
        +float avg_paragraphs_per_doc
        +float avg_words_per_doc
        +int min_words
        +int max_words
        +float median_words
        +dict schema
        +summary() str
    }

    class PostureTaxonomy {
        <<dataclass>>
        +Series all_postures
        +int COMMON_THRESHOLD = 100
        +int MODERATE_THRESHOLD = 10
        +common() list~str~
        +moderate() list~str~
        +rare() list~str~
        +singletons() list~str~
        +summary() dict
        +get_modeling_subset(min_samples: int) list~str~
        +to_dataframe() DataFrame
    }

    DatasetAnalyzer o-- DataLoader : loader (injected)
    DatasetAnalyzer ..> DatasetStatistics : creates
    DatasetAnalyzer ..> PostureTaxonomy : creates
```

### 4.2 Baseline Training Layer (`model_trainer.py`)

```mermaid
classDiagram
    class TextExtractor {
        <<utility>>
        +extract_text(sections: list)$ str
        +extract_all(df: DataFrame, col: str)$ list~str~
    }

    class DataPreparer {
        +DataLoader loader
        +float test_size = 0.15
        +float val_size = 0.15
        +int random_state = 42
        +Optional~int~ min_label_count
        -Optional~MultiLabelBinarizer~ _mlb
        -Optional~TfidfVectorizer~ _vectorizer
        +__init__(loader, test_size, val_size, random_state, min_label_count)
        -_get_viable_labels(df, posture_col) set~str~
        +prepare(text_col, posture_col, max_features, ngram_range) PreparedData
    }

    class PreparedData {
        <<dataclass>>
        +ndarray X_train
        +ndarray X_val
        +ndarray X_test
        +ndarray y_train
        +ndarray y_val
        +ndarray y_test
        +list~str~ train_texts
        +list~str~ val_texts
        +list~str~ test_texts
        +list~str~ label_names
        +Optional~MultiLabelBinarizer~ mlb
        +Optional~TfidfVectorizer~ vectorizer
        +summary() str
    }

    class BaselineTrainer {
        +float C = 1.0
        +int max_iter = 1000
        +Optional~OneVsRestClassifier~ model
        +__init__(C: float, max_iter: int)
        +train(X_train, y_train) BaselineTrainer
        +predict(X) ndarray
        +predict_proba(X) ndarray
    }

    DataPreparer o-- DataLoader : loader (injected)
    DataPreparer ..> TextExtractor : uses static methods
    DataPreparer ..> PreparedData : creates
    note for BaselineTrainer "Consumes X_train, y_train\n(raw ndarray from PreparedData)"
```

### 4.3 Transformer Training Layer (`bert_trainer.py`)

```mermaid
classDiagram
    class DeviceManager {
        <<utility>>
        +get_device(preference: str)$ torch.device
        +get_device_info()$ dict
    }

    class TrainingConfig {
        <<dataclass>>
        +int batch_size = 4
        +int gradient_accumulation_steps = 4
        +float learning_rate = 1e-5
        +int num_epochs = 5
        +float warmup_ratio = 0.1
        +float weight_decay = 0.01
        +float max_grad_norm = 1.0
        +int early_stopping_patience = 2
        +Optional~str~ checkpoint_dir
        +bool use_pos_weight = True
        +effective_batch_size() int
    }

    class TrainingHistory {
        <<dataclass>>
        +list~float~ train_losses
        +list~float~ val_losses
        +list~float~ val_f1_micro
        +list~float~ val_f1_macro
        +int best_epoch = 0
        +float best_f1 = 0.0
        +to_dict() dict
        +save(path: str) None
        +load(path: str)$ TrainingHistory
    }

    class SummaryCache {
        +Path cache_dir
        -dict _memory_cache
        +__init__(cache_dir: str)
        -_hash_text(text: str)$ str
        -_get_cache_path(text_hash: str) Path
        +get(text: str) Optional~str~
        +put(text: str, summary: str) None
        +get_stats() dict
    }

    class LegalSummarizer {
        +str DEFAULT_MODEL$
        +str model_name
        +int max_input_length
        +int summary_max_length
        +int summary_min_length
        +torch.device device
        +Optional~SummaryCache~ cache
        -_tokenizer
        -_model
        -_load_model() None
        +tokenizer
        +model
        +count_tokens(text: str) int
        +summarize(text: str, use_cache: bool) str
        +summarize_batch(texts, use_cache, show_progress, batch_size) list~str~
    }

    class Dataset {
        <<external>>
        torch.utils.data.Dataset
    }

    class LegalTextDataset {
        +list~str~ texts
        +ndarray labels
        +tokenizer
        +int max_length
        +__len__() int
        +__getitem__(idx: int) dict
    }

    Dataset <|-- LegalTextDataset : extends

    class LongDocumentStrategy {
        <<abstract>>
        +process(text, token_count, max_length)* str
    }

    class TruncateStrategy {
        +process(text, token_count, max_length) str
    }

    class HeadTailStrategy {
        +float head_ratio = 0.7
        +process(text, token_count, max_length) str
    }

    class SummarizeStrategy {
        -LegalSummarizer summarizer
        +process(text, token_count, max_length) str
    }

    class BaseTransformerTrainer {
        <<abstract>>
        +str model_name
        +int num_labels
        +int max_length
        +torch.device device
        -_tokenizer
        -_model
        -bool _is_trained
        -_loss_fn
        +TrainingHistory history
        +_create_model()* nn.Module
        -_load_tokenizer() AutoTokenizer
        -_compute_metrics(y_true, y_pred) dict
        -_training_step(batch, model) Tensor
        -_validation_step(val_loader, model) tuple~float, dict~
        +tokenizer
        +model nn.Module
        +count_tokens(text: str) int
        +count_tokens_batch(texts) list~int~
        +train(train_texts, y_train, val_texts, y_val, config, callback) BaseTransformerTrainer
        +predict_proba(texts, batch_size) ndarray
        +predict(texts, threshold) ndarray
        +save(path: str) None
        +load(path: str) BaseTransformerTrainer
    }

    class LegalLongformerTrainer {
        +str DEFAULT_MODEL$
        +__init__(num_labels, model_name, max_length, device)
        +_create_model() LongformerForSequenceClassification
    }

    class HybridLegalClassifier {
        +int num_labels
        +int max_length
        +str cache_dir
        +str device_name
        +LegalSummarizer summarizer
        +LegalLongformerTrainer classifier
        -dict _stats
        +tokenizer
        +history TrainingHistory
        +analyze_lengths(texts) DataFrame
        +get_length_stats(texts) dict
        +preprocess_texts(texts, show_progress) list~str~
        +train(train_texts, y_train, val_texts, y_val, config, preprocess, callback) HybridLegalClassifier
        +predict_proba(texts, preprocess, batch_size) ndarray
        +predict(texts, preprocess, threshold) ndarray
        +get_processing_stats() dict
        +save(path: str) None
        +load(path: str) HybridLegalClassifier
    }

    LongDocumentStrategy <|.. TruncateStrategy : realizes
    LongDocumentStrategy <|.. HeadTailStrategy : realizes
    LongDocumentStrategy <|.. SummarizeStrategy : realizes
    SummarizeStrategy --> LegalSummarizer : summarizer

    BaseTransformerTrainer <|-- LegalLongformerTrainer : extends
    BaseTransformerTrainer ..> TrainingConfig : configured by
    BaseTransformerTrainer *-- TrainingHistory : history
    BaseTransformerTrainer ..> LegalTextDataset : creates during train
    BaseTransformerTrainer ..> DeviceManager : uses

    LegalSummarizer *-- SummaryCache : cache
    LegalSummarizer ..> DeviceManager : uses

    HybridLegalClassifier *-- LegalSummarizer : summarizer
    HybridLegalClassifier *-- LegalLongformerTrainer : classifier
```

### 4.4 Evaluation Layer (`model_evaluator.py`)

```mermaid
classDiagram
    class MultiLabelEvaluator {
        +list~str~ label_names
        +__init__(label_names: list~str~)
        +evaluate(y_true, y_pred) EvaluationResults
        -_compute_per_class_metrics(y_true, y_pred) DataFrame
    }

    class EvaluationResults {
        <<dataclass>>
        +float f1_micro
        +float f1_macro
        +float f1_weighted
        +float f1_samples
        +float precision_micro
        +float precision_macro
        +float recall_micro
        +float recall_macro
        +float hamming_loss
        +float exact_match_ratio
        +DataFrame per_class_metrics
        +list~str~ label_names
        +summary() str
        +get_top_classes(n, metric) DataFrame
        +get_bottom_classes(n, metric) DataFrame
        +get_feasibility_analysis(human_kappa_low, human_kappa_high) DataFrame
    }

    class ModelComparison {
        <<dataclass>>
        +str model1_name
        +str model2_name
        +ndarray y_true
        +ndarray y_pred1
        +ndarray y_pred2
        +list~str~ label_names
        +agreement_matrix() dict
        +per_class_comparison() DataFrame
        +ensemble_predictions(strategy: str) ndarray
    }

    class compute_threshold_analysis {
        <<function>>
        (y_true, y_proba, label_names, thresholds) DataFrame
    }

    class save_predictions {
        <<function>>
        (y_pred, y_proba, filepath) None
    }

    class load_predictions {
        <<function>>
        (filepath) tuple~ndarray, ndarray~
    }

    class create_classification_report_df {
        <<function>>
        (y_true, y_pred, label_names) DataFrame
    }

    MultiLabelEvaluator ..> EvaluationResults : creates
    ModelComparison ..> MultiLabelEvaluator : used alongside
```

---

## 5. Sequence Diagrams

### 5.1 Data Discovery (Notebook 01)

```mermaid
sequenceDiagram
    actor User
    participant DL as DataLoader
    participant DA as DatasetAnalyzer
    participant PT as PostureTaxonomy
    participant VZ as Visualization

    User->>DL: DataLoader(file_path)
    User->>DL: load()
    DL-->>DL: pd.read_json(lines=True)
    DL-->>User: DataFrame (18,000 docs)

    User->>DL: get_schema()
    DL-->>User: {documentId, postures, sections}

    User->>DA: DatasetAnalyzer(loader)
    User->>DA: compute_statistics()
    DA->>DL: df (property access)
    DA-->>DA: _detect_columns()
    DA-->>DA: count paragraphs, words
    DA-->>DA: explode postures, value_counts
    DA-->>User: DatasetStatistics

    User->>DA: get_posture_taxonomy()
    DA-->>PT: PostureTaxonomy(posture_distribution)

    User->>PT: summary()
    PT-->>User: {common: 27, moderate: 57, rare: 140}

    User->>PT: get_modeling_subset(min_samples=50)
    PT-->>User: 41 viable labels (94.8% coverage)

    User->>DA: get_class_imbalance_ratio()
    DA-->>User: 9,197x (On Appeal vs rarest)

    User->>VZ: plot_posture_distribution(stats)
    VZ-->>User: Bar chart figure

    User->>VZ: plot_text_length_distribution(word_counts)
    VZ-->>User: Histogram figure

    User->>VZ: plot_class_imbalance(stats)
    VZ-->>User: Donut chart figure
```

### 5.2 Baseline Modeling (Notebook 02, Part 1)

```mermaid
sequenceDiagram
    actor User
    participant DL as DataLoader
    participant DP as DataPreparer
    participant TE as TextExtractor
    participant BT as BaselineTrainer
    participant EV as MultiLabelEvaluator

    User->>DL: DataLoader(file_path)
    User->>DP: DataPreparer(loader, min_label_count=50)

    User->>DP: prepare(max_features=10000, ngram_range=(1,2))
    DP->>DL: df (property)
    DP->>TE: extract_all(df, 'sections')
    TE-->>DP: list[str] (flat texts)
    DP-->>DP: _get_viable_labels() → 41 labels
    DP-->>DP: filter docs with <10 words
    DP-->>DP: MultiLabelBinarizer.fit_transform()
    DP-->>DP: train_test_split (70/15/15)
    DP-->>DP: TfidfVectorizer.fit_transform()
    DP-->>User: PreparedData

    User->>BT: BaselineTrainer(C=1.0)
    User->>BT: train(X_train, y_train)
    BT-->>BT: OneVsRestClassifier(LogisticRegression).fit()
    BT-->>User: trained model

    User->>BT: predict(X_test)
    BT-->>User: y_pred (binary matrix)

    User->>BT: predict_proba(X_test)
    BT-->>User: y_proba (probability matrix)

    User->>EV: MultiLabelEvaluator(label_names)
    User->>EV: evaluate(y_test, y_pred)
    EV-->>EV: f1_score, precision_score, recall_score
    EV-->>EV: _compute_per_class_metrics()
    EV-->>User: EvaluationResults (F1 Micro=0.752)

    User->>User: save_predictions(y_pred, y_proba, 'tfidf_test_predictions.npz')
```

### 5.3 Transformer Training (Notebook 02, Part 2)

```mermaid
sequenceDiagram
    actor User
    participant HC as HybridLegalClassifier
    participant LS as LegalSummarizer
    participant SC as SummaryCache
    participant LT as LegalLongformerTrainer
    participant DS as LegalTextDataset

    User->>HC: HybridLegalClassifier(num_labels=41, ...)
    HC->>LS: LegalSummarizer(model, cache_dir)
    LS->>SC: SummaryCache('outputs/summaries')
    HC->>LT: LegalLongformerTrainer(num_labels=41)

    User->>HC: get_length_stats(all_texts)
    HC->>LT: count_tokens_batch(texts)
    HC-->>User: {direct: 65.1%, summarize: 33.1%}

    User->>HC: preprocess_texts(train_texts)
    HC->>HC: analyze_lengths(texts)
    Note over HC: Identify long docs (>4096 tokens)

    HC->>LS: summarize_batch(long_texts, batch_size=8)
    Note over LS: Phase 1: Check cache for all texts
    loop For each text in batch
        LS->>SC: get(text)
        alt Cache hit
            SC-->>LS: cached summary
        else Cache miss
            LS-->>LS: collect for GPU inference
        end
    end
    Note over LS: Phase 2: Batch GPU inference on cache misses
    LS-->>LS: tokenizer(batch) + model.generate(batch)
    loop For each new summary
        LS->>SC: put(text, summary)
    end
    LS-->>HC: all summaries (cached + new)

    HC-->>User: processed texts (some summarized)

    User->>HC: train(texts, y_train, val_texts, y_val, config)
    HC->>LT: train(processed_texts, y_train, ...)
    LT->>DS: LegalTextDataset(texts, labels, tokenizer)

    loop For each epoch (1..5)
        LT-->>LT: _training_step(batch) per batch
        LT-->>LT: gradient accumulation (every 4 steps)
        LT-->>LT: _validation_step(val_loader)
        LT-->>LT: early stopping check
        Note over LT: Best epoch: 4 (val_f1=0.628)
    end

    LT-->>HC: trained model
    User->>HC: save('legal_longformer_best.pt')
    HC->>LT: save(path)
    LT-->>LT: torch.save(model_state_dict)
    LT-->>LT: history.save(.history.json)
```

### 5.4 Evaluation & Threshold Optimization (Notebook 03)

```mermaid
sequenceDiagram
    actor User
    participant HC as HybridLegalClassifier
    participant EV as MultiLabelEvaluator

    User->>HC: HybridLegalClassifier(num_labels=41)
    User->>HC: load('legal_longformer_best.pt')

    User->>HC: preprocess_texts(val_texts)
    Note over HC: Uses cached summaries (instant)
    User->>HC: preprocess_texts(test_texts)

    User->>HC: predict_proba(val_texts, preprocess=False)
    HC-->>User: y_proba_val (2509 x 41)

    User->>HC: predict_proba(test_texts, preprocess=False)
    HC-->>User: y_proba_test (2509 x 41)

    rect rgb(240, 248, 255)
        Note over User: Global Threshold Sweep (on VAL)
        loop threshold in [0.1, 0.2, ..., 0.9]
            User->>User: y_pred = (y_proba_val >= threshold)
            User->>EV: evaluate(y_val, y_pred)
            EV-->>User: F1 at each threshold
        end
        Note over User: Best global threshold = 0.6
    end

    rect rgb(255, 248, 240)
        Note over User: Per-Class Threshold Optimization (on VAL)
        loop For each of 41 classes
            User->>User: sweep thresholds [0.3..0.9]
            User->>User: select threshold maximizing F1 for this class
        end
        Note over User: 41 thresholds, range 0.45-0.90, mean 0.72
    end

    User->>User: Apply per-class thresholds to TEST
    User->>EV: evaluate(y_test, y_pred_perclass)
    EV-->>User: EvaluationResults (F1 Micro=0.774)

    Note over User,EV: results: EvaluationResults (returned by evaluate)
    User->>User: results.get_feasibility_analysis(0.63, 0.93)
    Note over User: Feasibility DataFrame (automation tiers)
```

### 5.5 Ensemble Analysis (Notebook 04)

```mermaid
sequenceDiagram
    actor User
    participant IO as load_predictions
    participant MC as ModelComparison
    participant EV as MultiLabelEvaluator

    User->>IO: load_predictions('tfidf_test_predictions.npz')
    IO-->>User: y_pred_tfidf, y_proba_tfidf

    User->>IO: load_predictions('legal_longformer_test_predictions.npz')
    IO-->>User: y_pred_lf, y_proba_lf

    User->>MC: ModelComparison('TF-IDF', 'Longformer', y_true, y_pred_tfidf, y_pred_lf, label_names)

    User->>MC: agreement_matrix()
    MC-->>User: {agreement: 96.6%, both_pos: 4.6%, only_tfidf: 0.6%, only_lf: 2.8%}

    User->>MC: per_class_comparison()
    MC-->>User: DataFrame (TF-IDF better: 21, LF better: 5, similar: 15)

    rect rgb(240, 255, 240)
        Note over User: Test Ensemble Strategies
        User->>MC: ensemble_predictions('union')
        MC-->>MC: np.maximum(y_pred1, y_pred2)
        MC-->>User: y_pred_union
        User->>EV: evaluate(y_test, y_pred_union)
        EV-->>User: F1 Micro = 0.618

        User->>MC: ensemble_predictions('intersection')
        MC-->>MC: np.minimum(y_pred1, y_pred2)
        MC-->>User: y_pred_intersection
        User->>EV: evaluate(y_test, y_pred_intersection)
        EV-->>User: F1 Micro = 0.783 ✓ Best
    end

    Note over User: Winner: Intersection (AND)\n+3.03pp over TF-IDF baseline
```

---

## 6. Results Summary

### 6.1 Model Performance (Test Set)

| Model | F1 Micro | F1 Macro | Precision | Recall |
|-------|----------|----------|-----------|--------|
| TF-IDF Baseline | 0.752 | 0.533 | 0.648 | 0.897 |
| LF Per-Class Thresholds | 0.774 | 0.600 | 0.712 | 0.847 |
| **Ensemble (AND)** | **0.783** | **0.575** | **0.713** | **0.868** |

### 6.2 Automation Feasibility Tiers

```mermaid
pie title Posture Automation Feasibility (41 classes)
    "Full Automation (F1 ≥ 0.93)" : 1
    "Assisted (0.63 ≤ F1 < 0.93)" : 17
    "Human Review (0.50 ≤ F1 < 0.63)" : 11
    "Manual Only (F1 < 0.50)" : 12
```

| Tier | Classes | Coverage | Action |
|------|---------|----------|--------|
| Full Automation | 1 (Appellate Review) | F1 = 0.95 | Deploy directly |
| Assisted Automation | 12–17 | F1 0.63–0.93 | Model suggests, human reviews |
| Human Review | ~11 | F1 0.50–0.63 | Model assists, human decides |
| Manual Only | ~12 | F1 < 0.50 | Insufficient model confidence |

> ~92% of labeling volume (by sample count) is covered by the automatable + review tiers.
