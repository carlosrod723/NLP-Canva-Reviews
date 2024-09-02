# Natural Language Processing (NLP) - Canva Reviews

## Overview
This repository contains a Natural Language Processing (NLP) project focused on sentiment analysis of user reviews for the Canva application. The project involves extensive text preprocessing and the development of a binary classification model to predict the sentiment of reviews as either positive or negative. The primary goal is to demonstrate the application of various NLP techniques, including tokenization, stopwords removal, stemming, and the creation of Bag of Words, N-grams, and TF-IDF representations, followed by the application of a Logistic Regression model for sentiment prediction.

## Table of Contents
- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Data Dictionary](#data-dictionary)
- [Tech Stack](#tech-stack)
- [Approach](#approach)
- [Execution Instructions](#execution-instructions)
- [Results and Analysis](#results-and-analysis)
- [Challenges and Considerations](#challenges-and-considerations)
- [Conclusion](#conclusion)
- [License](#license)

## Project Objectives
The objectives of this project are:
1. To understand the fundamentals of text preprocessing in NLP.
2. To build a binary classification model that can accurately predict the sentiment of user reviews.
3. To apply and compare various NLP techniques such as tokenization, stopwords removal, stemming, and vectorization methods.

## Dataset
The dataset used in this project contains over a thousand user reviews of the Canva application. Each review is labeled with a sentiment indicating whether it is positive or negative. Additional features provide metadata about the review, such as the user who wrote it and the version of the application being reviewed.

## Data Dictionary

| Column                   | Description                                                                 |
| -------------------------| --------------------------------------------------------------------------- |
| **review_id**              | Unique identifier for each review.                                          |
| **user_name**              | The name of the user who submitted the review.                              |
| **user_image**             | URL of the user's profile image.                                            |
| **review**                | The text content of the review.                                             |
| **score**                 | Rating score given by the user (e.g., 1 to 5 stars).                        |
| **thumbs_up_count**         | Number of users who found the review helpful.                               |
| **review_created_version**  | The version of the application at the time the review was written.          |
| **at**                    | Timestamp of when the review was written.                                   |
| **reply_content**          | The content of the reply from Canva to the review (if any).                 |
| **replied_at**             | Timestamp of when the reply was made by Canva.                              |
| **sentiment**             | Sentiment label indicating whether the review is positive or negative.      |
| **sub_category**          | Category to which the review belongs (if applicable).                       |
| **sub_category_test**     | Test category for validation purposes (if applicable).                      |

## Tech Stack
- **Language:** Python
- **Libraries:** pandas, seaborn, matplotlib, nltk, scikit-learn
- **Tools:** Google Colab, GitHub

## Approach
1. **Data Description and Visualization:**
   - Conduct initial data exploration to understand the distribution of reviews and sentiment.
   - Visualize key metrics using seaborn and matplotlib.

2. **Introduction to NLTK Library:**
   - Utilize the Natural Language Toolkit (nltk) for various NLP preprocessing tasks.

3. **Data Preprocessing:**
   - Convert all text to lowercase to maintain uniformity.
   - Tokenize the text into words using nltk's word_tokenize function.
   - Remove stopwords using nltk's predefined list of stopwords.
   - Remove punctuation from the text.
   - Apply stemming using both Porter and Lancaster stemmers to reduce words to their base forms.

4. **Bag of Words:**
   - Create Bag of Words representations using both binary and non-binary methods.
   - Explore the impact of N-grams on the model's performance.

5. **TF-IDF:**
   - Compute the Term Frequency-Inverse Document Frequency (TF-IDF) scores for each word in the corpus.

6. **Model Building and Accuracy:**
   - Build a Logistic Regression model to classify the sentiment of the reviews.
   - Evaluate the model's performance using accuracy metrics.

7. **Predictions on New Reviews:**
   - Test the model's predictions on new, unseen reviews to assess its generalization ability.
   - Examine the prediction probabilities to understand the model's confidence in its predictions.

## Results and Analysis
- **Model Performance:**
  - Summarize the results of the sentiment analysis model, highlighting the key findings and observations.
  - Evaluate the model using metrics such as accuracy, precision, recall, and F1-score to provide a comprehensive view of its performance.
  - Visualize the model's performance on the training and testing datasets using appropriate charts, such as confusion matrices, ROC curves, or precision-recall curves.
  
- **Accuracy:**
  - Discuss the accuracy achieved by the model on both the training and testing datasets.
  - Compare the accuracy against baseline models or previous iterations, if applicable.

- **Precision and Recall:**
  - Analyze the precision and recall metrics to understand the model's ability to correctly identify positive and negative sentiments.
  - Discuss any trade-offs observed between precision and recall, particularly in the context of the dataset's characteristics.

- **Model Insights:**
  - Provide insights into which features (e.g., specific words or phrases) contributed most to the model's predictions.
  - Explore any patterns or trends observed in the model's misclassifications, if relevant.

## Challenges and Considerations
- **Data Imbalance:**
  - The dataset may have an uneven distribution of positive and negative reviews, which could impact model performance.
  - Discuss strategies employed to address data imbalance, such as oversampling, undersampling, or using balanced class weights in the model.

- **Text Preprocessing:**
  - Ensuring effective text preprocessing is crucial for the model's accuracy.
  - Discuss the impact of different preprocessing steps (e.g., tokenization, stopwords removal, stemming) on the model's performance.
  - Mention any challenges faced during preprocessing, such as handling special characters, emojis, or non-standard text.

- **Model Generalization:**
  - Overfitting the model to training data can reduce its effectiveness on unseen data.
  - Discuss the steps taken to ensure the model generalizes well, such as using cross-validation, regularization techniques, or fine-tuning hyperparameters.
  - Address any challenges related to model generalization, including the trade-offs between model complexity and performance.

## Conclusion
The sentiment analysis of Canva reviews using NLP techniques was successful in demonstrating the effectiveness of various text preprocessing methods and classification models. The N-grams Logistic Regression model, in particular, outperformed others, achieving the highest accuracy on both the training and testing datasets. The TF-IDF representation also showed strong performance by focusing on word importance, while the binary Bag of Words model offered simplicity and robustness.

The final model was tested on new reviews, and its predictions were in line with the expected sentiments, further confirming its reliability. The examination of prediction probabilities provided additional insights into the model's confidence levels, ensuring that the predictions were both accurate and trustworthy.

The work here highlights the importance of careful feature engineering, model selection, and evaluation in building effective NLP models. The results obtained provide a solid foundation for future work in text classification and sentiment analysis.

## License
This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
