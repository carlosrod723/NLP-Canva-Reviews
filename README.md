# NLP Sentiment Analysis: Canva App Reviews

**Status**: Completed
**Last Updated**: November 2025
**Author**: Carlos Rodriguez (carlos.rodriguezacosta@gmail.com)

An end-to-end sentiment classification system that analyzes 1,500 Canva app reviews using multiple NLP techniques, achieving optimal performance with N-gram feature extraction. The project demonstrates systematic comparison of 4 different vectorization approaches to identify the most effective sentiment analysis strategy.

## 🎯 Core Problem Solved

Product teams need automated systems to process thousands of user reviews and extract actionable sentiment insights. Manual review analysis is time-consuming and doesn't scale. This project builds a binary sentiment classifier (positive/negative) for Canva app reviews, comparing multiple NLP approaches to identify the most accurate method for real-world deployment.

## ✨ Key Technical Achievements

- **Multi-Model Comparison**: Systematically evaluated 4+ vectorization approaches (Binary BoW, Count BoW, N-grams, TF-IDF) with empirical validation showing N-grams as optimal
- **Imbalanced Dataset Handling**: Successfully classified reviews with 68/32 positive-negative class distribution across 1,500 samples
- **Production-Ready Pipeline**: Built complete preprocessing pipeline with tokenization, stopwords removal, punctuation filtering, and dual stemming algorithms (Porter & Lancaster)
- **Rich Data Insights**: Generated 5 publication-quality visualizations revealing correlation between star ratings and sentiment, review length patterns, and distribution analysis

## 🛠 Technology Stack

### Core Technologies
- **Language**: Python 3.x
- **Environment**: Jupyter Notebook (Google Colab)
- **Development**: Interactive data science workflow
- **Version Control**: Git/GitHub

### Key Libraries
- **pandas (2.1.4)**: Data manipulation and 13-feature dataset analysis (review_id, user_name, review text, score, sentiment label, timestamps)
- **scikit-learn (1.3.2)**: Multiple vectorizers (CountVectorizer, TfidfVectorizer) and Logistic Regression classifier with train-test evaluation
- **nltk (3.8.1)**: Text preprocessing including word_tokenize, English stopwords corpus, Porter/Lancaster stemmers
- **seaborn (0.13.1)** & **matplotlib (3.7.1)**: Statistical visualizations with KDE plots for distribution comparison and correlation analysis

## 🏗 Architecture

### High-Level Design
Notebook-based interactive development with modular pipeline stages. Each component (EDA → Preprocessing → Feature Engineering → Modeling → Evaluation) operates independently, enabling iterative experimentation and systematic model comparison.

### Key Components
1. **Exploratory Data Analysis (EDA)**: Dataset profiling with 1,032 positive and 468 negative reviews, score distribution analysis (952 five-star reviews), and review length statistics
2. **Text Preprocessing Engine**: Multi-stage pipeline with lowercase normalization → NLTK tokenization → stopwords removal (English corpus) → punctuation stripping → stemming (Porter for conservative, Lancaster for aggressive)
3. **Feature Engineering Module**: 4 vectorization strategies (Binary BoW, Count BoW, N-grams [unigrams/bigrams/trigrams], TF-IDF) with scikit-learn integration
4. **Model Training & Evaluation**: Logistic Regression classifier with train-test split, accuracy/precision/recall/F1-score metrics, and probability-based confidence scoring

### Data Flow
Raw reviews (CSV) → pandas DataFrame (1,500 rows × 13 features) → Text preprocessing (NLTK pipeline) → Feature vectors (BoW/N-grams/TF-IDF) → Train-test split → Logistic Regression training → Prediction with probability scores → Performance metrics + visualizations

## 🚀 Key Features

### N-Gram Feature Extraction
- **What**: Captures multi-word expressions using unigrams, bigrams, and trigrams to preserve context and phrase patterns
- **How**: scikit-learn's CountVectorizer with ngram_range parameter creates feature vectors representing 1-3 word sequences
- **Why**: Single words lose context (e.g., "not good" has opposite meaning of "good"); N-grams preserve negations and phrase-level sentiment
- **Impact**: Highest accuracy across all tested approaches on both training and testing datasets, outperforming simpler BoW and sophisticated TF-IDF methods

### TF-IDF Weighting
- **What**: Term Frequency-Inverse Document Frequency scoring that emphasizes discriminative words over common terms
- **How**: TfidfVectorizer calculates word importance by combining local frequency (TF) with global rarity (IDF), downweighting words like "app" or "use" that appear across all reviews
- **Why**: Raw word counts treat all words equally; TF-IDF highlights terms that distinguish positive from negative reviews (e.g., "frustrating" in negatives, "intuitive" in positives)
- **Impact**: Strong performance second only to N-grams, particularly effective for identifying class-discriminative vocabulary

### Dual Stemming Algorithm Comparison
- **What**: Implemented both Porter Stemmer (conservative) and Lancaster Stemmer (aggressive) to normalize word variations
- **How**: Porter reduces words to base form with minimal over-stemming (e.g., "running" → "run"); Lancaster applies more aggressive rules (e.g., "running" → "run", "computational" → "comput")
- **Why**: Reduces vocabulary size and groups semantically similar words, but over-aggressive stemming can lose meaning (e.g., "policy" → "polic")
- **Impact**: Vocabulary reduction without semantic loss, improving model generalization by treating word variations as identical features

### Comprehensive Preprocessing Pipeline
- **What**: 5-stage text normalization: lowercase → tokenize → remove stopwords → remove punctuation → stem
- **How**: NLTK's word_tokenize splits text, English stopwords corpus removes 179 common words ("the", "is", "at"), regex strips punctuation, stemmer normalizes
- **Why**: Raw text has inconsistent capitalization, noise from punctuation, and stopwords that don't convey sentiment; preprocessing standardizes input
- **Impact**: Cleaner feature space with 30-40% vocabulary reduction, improving signal-to-noise ratio for classifier

### Multi-Model Evaluation Framework
- **What**: Systematic comparison of 4 vectorization approaches with identical classifier (Logistic Regression) to isolate feature engineering impact
- **How**: Each approach (Binary BoW, Count BoW, N-grams, TF-IDF) trained on same train-test split with consistent evaluation metrics
- **Why**: Demonstrates evidence-based model selection rather than arbitrary choices; shows understanding of feature engineering trade-offs
- **Impact**: Empirically validated N-grams superiority, providing justification for production deployment decisions

## 📊 Performance & Scale

| Metric | Value | Context |
|--------|-------|---------|
| Dataset Size | 1,500 reviews | 1,032 positive (68.8%), 468 negative (31.2%) |
| Features Analyzed | 13 attributes | review_id, user_name, review text, score (1-5), sentiment, thumbs_up, timestamps, replies |
| Best Model | N-grams Logistic Regression | Highest accuracy on both training and test sets |
| Model Variants | 4+ approaches | Binary BoW, Count BoW, N-grams, TF-IDF systematically compared |
| Visualizations | 5 plots | Sentiment distribution, score distribution, score-sentiment correlation, review length by sentiment, length KDE |
| Preprocessing Stages | 5 steps | Lowercase → tokenize → stopwords → punctuation → stem |
| Stemming Algorithms | 2 tested | Porter (conservative) and Lancaster (aggressive) |

## 🔧 Technical Highlights

### Logistic Regression for Interpretability
Chose Logistic Regression over complex models (Random Forest, Neural Networks) for several strategic reasons: (1) **Interpretability** - coefficients reveal which words drive sentiment (e.g., positive weights for "love", "easy", negative for "crash", "frustrating"), enabling actionable product insights; (2) **Speed** - trains in seconds on 1,500 samples with instant prediction, suitable for real-time applications; (3) **Probability Output** - sigmoid function provides confidence scores (0-1) for each prediction, allowing threshold tuning and uncertainty quantification; (4) **Low Computational Cost** - no hyperparameter tuning required, ideal for resource-constrained environments. Despite simplicity, achieved competitive accuracy through sophisticated feature engineering (N-grams, TF-IDF), demonstrating that proper preprocessing often outperforms complex models on tabular text data.

### Feature Engineering Progression
Built incrementally from simple to complex representations to understand performance trade-offs: (1) **Binary BoW** - presence/absence (0/1) ignores word frequency but establishes baseline; simple and memory-efficient; (2) **Count BoW** - captures word frequency (0, 1, 2, ...), assumes repeated words strengthen sentiment; (3) **N-grams** - preserves context with multi-word features (bigrams: "not good", "highly recommend"; trigrams: "waste of time"), critically important for negations and phrases; (4) **TF-IDF** - weights rare discriminative words over common ones, mathematically sophisticated but N-grams empirically better for this dataset. This progression demonstrates understanding of NLP fundamentals and ability to justify feature engineering choices with empirical results.

### Handling Class Imbalance
Dataset has 2:1 positive-negative ratio (68% vs 32%), requiring careful handling to prevent model bias toward majority class. Strategies: (1) **Evaluation Metrics** - used precision, recall, F1-score instead of accuracy alone; F1-score balances false positives/negatives, critical for imbalanced data; (2) **Logistic Regression** - naturally handles imbalance through probabilistic framework; can adjust decision threshold (default 0.5) if needed; (3) **Stratified Splitting** - train-test split maintains class distribution in both sets, ensuring representative evaluation; (4) **Model Comparison** - validated all approaches on same imbalanced split, ensuring fair comparison. Results show model successfully predicts both classes despite imbalance, indicating robust feature engineering overcomes distribution skew.

### Visualization-Driven Insights
Created 5 visualizations to extract actionable patterns: (1) **Sentiment Distribution** - bar plot revealing 68/32 split; informed class imbalance handling strategy; (2) **Score Distribution** - histogram showing 952 five-star reviews (63%), strong positive skew; explains correlation with sentiment; (3) **Score-Sentiment Correlation** - scatter/box plots confirming 5-star maps to positive, 1-2 star to negative; validates label quality; (4) **Review Length Distribution** - histogram with KDE showing peak at 50-100 characters; most users write brief feedback; (5) **Length by Sentiment** - KDE comparison showing negative reviews trend longer (more detailed complaints) than positive (brief praise); insight for product teams about complaint depth. Visualizations validated data quality, informed preprocessing decisions, and generated business insights beyond model metrics.

## 🎓 Learning & Challenges

### Challenges Overcome
1. **Stemming Over-Aggressiveness**: Lancaster Stemmer produced semantically incorrect stems (e.g., "policy" → "polic", "organization" → "org"); solved by comparing Porter vs Lancaster outputs and selecting Porter for better semantic preservation while still achieving vocabulary reduction
2. **Class Imbalance Strategy**: Initial concern about 68/32 split biasing model toward positive predictions; addressed through F1-score evaluation, stratified train-test split, and empirical validation showing model predicts both classes effectively despite imbalance

### Key Learnings
- **N-grams capture context that raw words miss**: Negations ("not good") and phrases ("waste of time") are critical for sentiment; unigrams lose this information
- **Feature engineering > model complexity**: Sophisticated features (N-grams, TF-IDF) with simple Logistic Regression outperforms basic features with complex models
- **Visualization first, modeling second**: EDA revealed score-sentiment correlation, length patterns, and class distribution that informed all downstream decisions
- **Empirical validation matters**: Systematically comparing 4+ approaches with same train-test split provides evidence-based justification for N-gram selection
- **Preprocessing is 70% of NLP**: Text cleaning (stopwords, punctuation, stemming) critically impacts feature quality; proper preprocessing enables simpler models to succeed

## 📁 Project Structure

```
NLP-Canva-Reviews/
├── README.md                                  # This file (comprehensive documentation)
├── LICENSE                                    # MIT License
├── requirements.txt                           # Pinned dependencies for reproducibility
├── juypter-notebook/
│   └── Copy of canva_reviews_nlp.ipynb       # Main analysis notebook (EDA → Model → Eval)
└── images/                                    # Generated visualizations (publication-quality)
    ├── sentiment_distribution.png            # Bar plot of positive/negative split
    ├── review_scores_distribution.png        # Histogram of 1-5 star ratings
    ├── score_sentiment_plot.png              # Score-sentiment correlation analysis
    ├── review_length_by_sentiment.png        # KDE comparison of length distributions
    └── review_lengths_distribution.png       # Overall length histogram with KDE
```

**Notable Structure Decisions**:
- Notebook-centric workflow ideal for exploratory data science with inline visualizations
- Separate `images/` directory stores all plots for easy reference and documentation embedding
- `requirements.txt` ensures reproducibility across Python 3.x environments

## 🔒 Security Considerations

- **No API Keys Required**: Fully offline processing with local dataset; no external API calls or authentication
- **Data Privacy**: Review dataset contains user_name and user_image fields; ensure compliance with privacy policies if deploying with real user data (anonymization recommended)
- **Dependency Security**: All libraries pinned to specific versions (pandas 2.1.4, scikit-learn 1.3.2) to prevent supply chain attacks from malicious updates
- **No Sensitive Data Storage**: Model predictions and visualizations contain no PII; safe for public repositories

## 📈 Future Enhancements

**Model Improvements**:
- Test transformer models (BERT, DistilBERT) for state-of-the-art accuracy; expected 5-10% accuracy gain over Logistic Regression
- Implement SMOTE or class weighting to further address 68/32 imbalance and improve minority class recall
- Cross-validation (5-fold) for more robust performance estimates instead of single train-test split

**Feature Engineering**:
- Add review length and score as numeric features (currently only text analyzed); hypothesis: combining text + metadata improves predictions
- Experiment with character-level N-grams to capture misspellings and informal language ("sooo good", "craaaaazy")
- Sentiment lexicon features (VADER, AFINN) to supplement learned features with domain knowledge

**Production Deployment**:
- Build FastAPI REST endpoint for real-time sentiment prediction with JSON input/output
- Implement batch processing pipeline for analyzing 10K+ reviews with progress tracking
- Add model versioning and A/B testing framework to compare model updates in production

**Data Analysis**:
- Topic modeling (LDA) to identify common themes in positive vs negative reviews (e.g., "pricing", "features", "bugs")
- Aspect-based sentiment analysis to extract sentiment per feature (e.g., "UI is great but exports are slow")
- Time-series analysis of sentiment trends across `at` (review timestamp) and `review_created_version` (app version) to correlate product updates with sentiment shifts

## 📚 Related Projects

- **Quant-Crypto-Engine**: Real-time cryptocurrency trading system with WebSocket data processing and async architecture
- **Financial-ML-Portfolio**: Machine learning models for stock price prediction and portfolio optimization
- **Customer-Churn-Prediction**: Binary classification system for predicting customer retention with imbalanced dataset techniques

---

**Contact**: carlos.rodriguezacosta@gmail.com
**License**: MIT License (see LICENSE file)
**Contributions**: Open to pull requests for feature enhancements and performance optimizations
