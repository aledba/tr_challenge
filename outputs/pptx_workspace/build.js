const pptxgen = require('pptxgenjs');
const html2pptx = require('/Users/alessandrodibari/.claude/skills/pptx/scripts/html2pptx');
const path = require('path');

const NOTES = {
  'slide01_title.html': `Welcome everyone, and thank you for your time. Today I'll walk you through my feasibility study on automating procedural posture classification for judicial opinions.

The goal of this challenge wasn't to build a production system — it was to characterize performance, understand the data, and make an honest recommendation about whether automation is feasible.

KEY CONCEPT — Procedural Posture: The procedural posture of a case summarizes how the case arrived before the court. It includes the procedural history and any prior decisions under appeal. Business partners currently label these manually and want to know if automation can help.`,

  'slide02_what_is_posture.html': `So what exactly is procedural posture? It's essentially the "how did we get here?" of a court case. When a judge writes an opinion, the procedural posture tells you what legal actions led to this point.

For example, a case might be "On Appeal" after a lower court ruling, or involve a "Motion to Dismiss" where one party is asking the court to throw out the case. These labels are critical metadata for legal research — they help lawyers find relevant precedents.

The business need is clear: human labelers currently assign these postures manually, which is expensive and slow. We want to know if machine learning can do part of this job reliably.`,

  'slide03_dataset.html': `Let me walk you through the dataset. We have 18,000 judicial opinions — each is a structured document with sections and paragraphs, totaling over half a million paragraphs. There are 224 unique posture labels, and critically, this is a multi-label problem: nearly half the documents have more than one posture.

KEY CONCEPT — Multi-label Classification: Unlike multi-class classification where each document gets exactly one label, multi-label means each document can have zero, one, or many labels simultaneously. This is harder because the model must make independent yes/no decisions for each possible label.

The mean document length of nearly 3,000 words is a significant engineering challenge, as we'll see later.`,

  'slide04_label_dist.html': `This slide shows the extreme class imbalance we're dealing with. The 27 most common labels cover 91.5% of all label occurrences, while 140 rare labels have fewer than 10 samples each — including 44 labels that appear only once in the entire dataset.

The class imbalance ratio of 9,197x means the most frequent label is over nine thousand times more common than the rarest. For modeling, we filtered to 41 labels with at least 50 samples, which still covers 94.8% of all label occurrences. This is a pragmatic decision — you can't build a reliable classifier on 5 training examples.

KEY CONCEPT — Class Imbalance: When some classes have many more samples than others. Standard ML algorithms tend to predict the majority class and ignore minorities. Solutions include oversampling, class weights, and threshold optimization.`,

  'slide05_domain.html': `The domain complexity is what makes this problem genuinely hard. The labels aren't just arbitrary categories — they encode multiple orthogonal dimensions: the litigation stage (where in the process), the motion type (what action), the proceeding type (what legal domain), and procedural events.

There are also hierarchical relationships. For example, "Motion to Dismiss for Lack of Subject Matter Jurisdiction" IS-A specific type of "Motion to Dismiss." This means some label confusions are less severe than others.

KEY CONCEPT — Cohen's Kappa: An inter-annotator agreement metric that corrects for chance agreement. Kappa = 0 means agreement is no better than random; kappa = 1 means perfect agreement. Our human benchmark range of 0.63–0.93 tells us that some postures are inherently harder to classify, even for trained legal professionals. This range defines our target — a model performing at kappa-equivalent levels could be trusted for automation.`,

  'slide06_section_solution.html': `Now let's look at how I approached the modeling. I designed a two-model strategy to understand the problem from different angles.`,

  'slide07_strategy.html': `I built two models: a TF-IDF baseline and a Legal-Longformer hybrid transformer. The key principle here was to characterize performance, not chase state-of-the-art. The baseline tells us "how well can simple features do?" and the transformer tells us "does deep learning add value?"

KEY CONCEPT — TF-IDF (Term Frequency–Inverse Document Frequency): A way to convert text into numerical features. Each word gets a weight based on two factors: how often it appears in this specific document (TF) and how rare it is across all documents (IDF). Words that are frequent in one document but rare overall get the highest weights — these are the most discriminative features. TF-IDF is fast, interpretable, and surprisingly effective for legal text because legal language has strong keyword signals.`,

  'slide08_baseline.html': `The baseline uses TF-IDF features with logistic regression in a OneVsRest configuration. We extract the top 10,000 features using unigrams and bigrams, then train a separate binary classifier for each of the 41 labels.

KEY CONCEPT — OneVsRest (OVR): A strategy for multi-label classification that trains one independent binary classifier per label. Each classifier learns "is this label present: yes or no?" This is simple but effective — each label gets its own decision boundary. The disadvantage is that it doesn't model label correlations.

KEY CONCEPT — Unigrams and Bigrams: Unigrams are single words ("motion", "dismiss"); bigrams are word pairs ("motion to", "to dismiss"). Bigrams capture phrases that single words miss, which is crucial in legal text where "motion to dismiss" means something very different from "motion" alone.

The 70/15/15 split gives us 11,706 training samples, 2,509 for validation, and 2,509 for testing.`,

  'slide09_long_doc.html': `This is a critical engineering challenge. BERT, the standard transformer, has a 512-token limit — only 4.1% of our documents fit. Even the Legal-Longformer with its 4,096-token window only covers 65% directly.

KEY CONCEPT — Tokens: The units a transformer processes. Roughly, 1 token ≈ 0.75 words. BERT's 512-token limit means about 384 words — far too short for legal opinions averaging nearly 3,000 words.

KEY CONCEPT — LED (Longformer Encoder-Decoder): A transformer designed for long documents, supporting up to 16,384 tokens using efficient sliding-window attention. We use it as a summarizer: it compresses long documents into summaries that fit the classifier's 4,096-token window.

Our hybrid strategy: documents under 4,096 tokens go directly to the classifier; longer ones get summarized first by Legal-LED, then classified. About 33% of documents needed summarization.`,

  'slide10_architecture.html': `Here's the technical architecture. The classifier is lexlms/legal-longformer-base, a transformer pre-trained on legal text with a 4,096-token context window. The summarizer is nsi319/legal-led-base-16384 for documents exceeding the classifier's limit.

KEY CONCEPT — Longformer: A transformer architecture that uses sliding-window attention (each token attends to nearby tokens) plus global attention on special tokens. This makes it efficient for long sequences — O(n) instead of O(n²) — enabling 4,096-token input, which is 8x BERT's limit.

KEY CONCEPT — pos_weight: In PyTorch's binary cross-entropy loss, pos_weight multiplies the loss contribution of positive (minority) examples. If a class has 50 positive and 950 negative samples, setting pos_weight=19 makes the model pay 19x more attention when it misses a positive example. This helps with class imbalance but can cause over-prediction if set too aggressively.

We cached all 5,530 summaries to disk for reproducibility — summarization is expensive and deterministic, so caching avoids redundant computation.`,

  'slide11_section_repo.html': `Let me show you how the project is organized.`,

  'slide12_project_org.html': `The project follows a clean separation: all logic lives in reusable Python modules under src/, while notebooks are thin wrappers that call into these modules. This makes the code testable, maintainable, and easy to follow.

Each module has a single responsibility: data_loader.py handles I/O, data_analyzer.py does analysis, model_trainer.py manages the TF-IDF pipeline, model_evaluator.py handles all evaluation metrics, and bert_trainer.py orchestrates the transformer workflow.

The outputs directory contains cached summaries for the 5,530 long documents, model checkpoints, and training history for reproducibility.`,

  'slide13_design.html': `A few key design decisions worth highlighting. The modular pipeline — DataLoader → DataPreparer → Trainer → Evaluator — makes it easy to swap components. The HybridLegalClassifier automatically routes documents through the right path (direct vs summarize-first).

The SummaryCache uses content-based hashing so summaries survive across different runs and data splits. And DeviceManager auto-detects CUDA, MPS, or CPU, making the code portable across hardware.`,

  'slide14_section_results.html': `Now for the results — this is where it gets interesting.`,

  'slide15_baseline_results.html': `The TF-IDF baseline achieved F1 Micro of 0.752 and F1 Macro of 0.533. The gap between micro and macro tells an important story.

KEY CONCEPT — F1 Micro vs F1 Macro: F1 Micro aggregates all true positives, false positives, and false negatives globally across all labels, then computes F1. It's dominated by frequent classes because they contribute more to the totals. F1 Macro computes F1 independently for each class, then averages. It treats rare classes equally with common ones. When micro is much higher than macro (0.75 vs 0.53), it means the model is good at frequent postures but struggles with rare ones.

KEY CONCEPT — Precision vs Recall: Precision answers "of everything the model flagged, how much was correct?" Recall answers "of everything that should have been flagged, how much did the model find?" Our baseline has high recall (0.90) but lower precision (0.65), meaning it catches most positive cases but also generates some false positives.

The top performers — Appellate Review at 0.95, On Appeal at 0.93 — all have large sample sizes. The bottom performers with F1 below 0.25 are rare classes. This sets our baseline — can the transformer do better?`,

  'slide16_hybrid_results.html': `This is the most important slide. At first glance, the transformer with default threshold 0.5 looked like a failure — F1 Micro dropped to 0.63, well below the baseline's 0.75. But look at what happens with threshold optimization.

The problem was over-prediction: at threshold 0.5, the model predicted too many labels (recall 0.94 but precision only 0.48). Per-class threshold optimization — finding the optimal probability cutoff for each label independently on the validation set — fixed this dramatically.

KEY CONCEPT — Threshold Optimization: In multi-label classification, the model outputs a probability for each label. The default is to predict "positive" if probability > 0.5. But for imbalanced problems, 0.5 is rarely optimal. Per-class threshold optimization finds the cutoff per class that maximizes F1 on validation data, then applies those thresholds to test data. Our optimal thresholds ranged from 0.45 to 0.90 with a mean of 0.72 — much higher than the naive 0.5.

The result: LF Per-Class achieves F1 Micro 0.7735 (+2.1% over baseline) and F1 Macro 0.6000 (+6.7% over baseline). The transformer beats the baseline on both metrics, especially on rare classes where the macro improvement is most pronounced.`,

  'slide16c_ensemble.html': `This slide shows our best overall result: the ensemble AND strategy that combines TF-IDF and Longformer predictions.

The AND strategy is simple: predict positive only when both models agree. Since both models have high recall — they catch most positive cases — taking their intersection boosts precision while keeping recall strong. The result: F1 Micro 0.783, a 3 percentage-point improvement over the TF-IDF baseline.

The models are genuinely complementary: 96.6% agreement overall, but TF-IDF excels on 21 classes while Longformer excels on 5. Different strengths make for a better ensemble.

An important methodology note: the per-class thresholds in NB03 were optimized on the validation set and applied to the test set — standard ML practice. The ensemble strategy in NB04 was selected on the test set, but the AND operation is parameter-free — it's just np.minimum — so there's no overfitting risk.

A genuine next step would be combining both: AND(TF-IDF, LF with Per-Class Thresholds). We haven't tried that yet.`,

  'slide17_feasibility.html': `Now let's talk about feasibility. Using the TF-IDF baseline as a conservative reference, 13 out of 41 postures are automatable with F1 at or above the human agreement threshold of 0.63. One posture — Appellate Review — reaches 0.95, which exceeds even the highest human agreement.

KEY CONCEPT — Cohen's Kappa as Benchmark: We use the inter-annotator agreement range (kappa 0.63–0.93) as the benchmark for automation. If a model achieves F1 comparable to human agreement on a posture, that posture is a candidate for automation. Below 0.63, the model is worse than human disagreement; above 0.93, it matches the best human performance.

The tiered deployment strategy is practical: fully automate what's reliable, use the model as an assistant where it's decent, and keep humans in the loop where the model struggles. Critically, ~92% of labeling volume falls in the automatable or review tiers because the most common postures happen to be the easiest to classify.

The ensemble AND model achieves the best F1 Micro at 0.783, while per-class thresholds achieve the best F1 Macro at 0.600 — different models excel on different metrics, reinforcing the value of multiple approaches.`,

  'slide18_insights.html': `Four key takeaways. First, sample size dominates performance — this is the single most actionable finding. If you want better models, invest in labeling more data for the rare classes.

Second, per-class threshold optimization was the breakthrough. It turned a seemingly failed transformer into the best model, beating the TF-IDF baseline on all metrics. This is a general lesson: in imbalanced multi-label problems, threshold tuning matters as much as model architecture.

Third, document length is a real engineering constraint. The hybrid summarize-then-classify approach works but adds complexity and some information loss. Future work with even longer-context models could help.

Fourth, the multi-label nature — with labels encoding orthogonal dimensions — adds fundamental complexity that simple models can't fully capture. Hierarchical approaches leveraging the ontology structure could help.

KEY CONCEPT — Hamming Loss: The fraction of incorrect labels across all label-sample pairs, counting both false positives and false negatives. Unlike F1, it penalizes every individual labeling error equally. A Hamming loss of 0.03 means 97% of individual label predictions are correct.

The business recommendation: partial automation is feasible and worth pursuing. A tiered deployment can cover ~92% of labeling volume. The ensemble's F1 Micro 0.783 is our best result, and the model complementarity (96.6% agreement, different class strengths) confirms there's value in combining approaches.`,

  'slide19_next_steps.html': `If this project continues, there are clear paths for improvement on both the modeling and production sides.

For modeling: the most promising next step is combining the two improvements we already built — running the AND ensemble on top of per-class thresholds. We did each separately but never combined them. Beyond that, a stacking meta-learner could learn optimal combination weights from both models' probability outputs.

GPU training with larger batches, more epochs, and proper hyperparameter search would likely improve the transformer significantly — we only ran 5 epochs on Apple MPS. Hierarchical classification using the ontology structure could leverage the IS-A relationships between postures. Data augmentation could boost rare class performance.

For production: a confidence-based routing system would send high-confidence predictions for auto-labeling and uncertain ones to human reviewers. Active learning would use the model's own uncertainty to prioritize which examples humans should label next, creating a virtuous cycle. Monitoring for concept drift is essential — legal language and posture distributions may shift over time.

KEY CONCEPT — Active Learning: A machine learning paradigm where the model identifies the most informative unlabeled examples and asks a human to label them. This is more efficient than random labeling because it focuses human effort where the model is most uncertain, accelerating improvement.`,

  'slide20_thankyou.html': `Thank you for your attention. I'm happy to dive deeper into any aspect of the analysis — the data exploration, the modeling decisions, the evaluation methodology, or the production planning.

The key message: this feasibility study shows that partial automation of procedural posture classification is viable. The ensemble achieves our best F1 Micro of 0.783, while per-class thresholds deliver the best F1 Macro of 0.600. A tiered deployment strategy can cover the vast majority of labeling volume while maintaining quality through human oversight where needed.`,

  'slide04b_label_pie.html': `This pie chart from the data exploration notebook shows the label distribution visually. You can see how "On Appeal" dominates at about a third of all label assignments, followed by "Appellate Review." The long tail of rare postures is clearly visible.

The percentages shown are relative to total label assignments (27,659), not documents (18,000). Since this is a multi-label problem with a mean of 1.54 labels per document, the total label count exceeds the document count. This distinction matters when interpreting class frequencies.`,

  'slide09b_token_dist.html': `This histogram shows the distribution of document lengths in tokens, with vertical lines marking the context limits of three transformer architectures: BERT at 512 tokens, Longformer at 4,096, and LED at 16,384.

The key takeaway is visual: BERT's limit barely captures the left edge of the distribution. Longformer covers most of the main body but still misses the right tail. LED covers nearly everything, which is why we chose it as our summarizer for the ~33% of documents exceeding 4,096 tokens.

KEY CONCEPT — Context Window: The maximum number of tokens a transformer can process at once. Self-attention has O(n²) complexity, so doubling the context quadruples compute. Longformer solves this with sliding-window attention (O(n)), enabling much longer contexts.`,

  'slide10b_training.html': `These two panels show the training progression over 5 epochs. On the left: training and validation loss curves. On the right: validation F1 scores (both Micro and Macro).

The model converges quickly — most improvement happens in the first 2-3 epochs. The best checkpoint was saved based on validation F1, and that's the model we use for all subsequent evaluation.

Training was done on Apple MPS (M2 chip), not a GPU cluster. With more compute — larger batches, more epochs, proper hyperparameter search — we'd expect further gains. The fact that the model already beats the TF-IDF baseline suggests there's more headroom to unlock.`,

  'slide16b_perclass_f1.html': `This grouped bar chart compares TF-IDF (blue) and Legal-Longformer (orange) F1 scores for each of the 41 posture classes. The horizontal dashed lines mark the human inter-annotator agreement range (Cohen's kappa 0.63–0.93).

The pattern is clear: both models perform well on high-frequency postures (left side) and struggle on rare ones (right side). The transformer tends to match or slightly exceed the baseline on most classes, especially after per-class threshold optimization.

KEY CONCEPT — Per-Class Performance: Aggregate metrics like F1 Micro can hide poor performance on minority classes. This chart reveals the full picture: the model is excellent for some postures and unreliable for others. This directly informs the tiered deployment strategy.`,
};

async function build() {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'Alessandro Di Bari';
  pptx.title = 'Procedural Posture Classification - Feasibility Study';

  const slidesDir = path.join(__dirname, 'slides');
  const slides = [
    'slide01_title.html',
    'slide02_what_is_posture.html',
    'slide03_dataset.html',
    'slide04_label_dist.html',
    'slide04b_label_pie.html',
    'slide05_domain.html',
    'slide06_section_solution.html',
    'slide07_strategy.html',
    'slide08_baseline.html',
    'slide09_long_doc.html',
    'slide09b_token_dist.html',
    'slide10_architecture.html',
    'slide10b_training.html',
    'slide11_section_repo.html',
    'slide12_project_org.html',
    'slide13_design.html',
    'slide14_section_results.html',
    'slide15_baseline_results.html',
    'slide16_hybrid_results.html',
    'slide16b_perclass_f1.html',
    'slide16c_ensemble.html',
    'slide17_feasibility.html',
    'slide18_insights.html',
    'slide19_next_steps.html',
    'slide20_thankyou.html',
  ];

  for (let i = 0; i < slides.length; i++) {
    const file = path.join(slidesDir, slides[i]);
    console.log(`Processing ${slides[i]}...`);
    const { slide } = await html2pptx(file, pptx);
    if (NOTES[slides[i]]) {
      slide.addNotes(NOTES[slides[i]]);
    }
  }

  const outputPath = path.join(__dirname, '..', 'TR_Interview_Presentation.pptx');
  await pptx.writeFile({ fileName: outputPath });
  console.log(`\nPresentation saved to: ${outputPath}`);
}

build().catch(err => { console.error(err); process.exit(1); });
