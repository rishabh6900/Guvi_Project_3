# Guvi_Project_3

# Overview
This project is a comprehensive sentiment analysis system for movie reviews, built using Python and machine learning techniques. The notebook processes the IMDB movie review dataset, performs text preprocessing, and implements multiple machine learning models to classify reviews as positive or negative.

Features
Data Analysis: Exploratory analysis of the IMDB review dataset with visualizations

# Text Preprocessing:

HTML tag removal

URL removal

punctuation

Text normalization (lowercasing)

Remove chat_words

TF-IDF

# Multiple ML Models: Implementation and comparison of four classification algorithms:

Logistic Regression

Random Forest

Multinomial Naive Bayes

Linear Support Vector Classifier (LinearSVC)

# Performance Evaluation: Comprehensive metrics including accuracy, precision, recall, and F1-score

# Dataset
The project uses the IMDB Dataset containing 50,000 movie reviews labeled as either "positive" or "negative". The dataset is balanced with an equal distribution of both sentiment classes.

Installation & Dependencies
To run this notebook, you'll need to install the following packages:

bash
pip install textblob pandas numpy scikit-learn matplotlib seaborn
Project Structure
Data Loading & Exploration: Loads the dataset and provides basic statistics and visualizations

Text Preprocessing: Cleans and prepares text data for analysis

Feature Extraction: Uses TF-IDF vectorization to convert text to numerical features

Model Training: Implements and trains four different classification models

Evaluation: Compares model performance using multiple metrics

Results Analysis: Provides insights into model performance and effectiveness

# Models Implemented
The notebook compares the performance of:

Logistic Regression: A linear model for binary classification

Random Forest: An ensemble method using multiple decision trees

Multinomial Naive Bayes: A probabilistic classifier based on Bayes' theorem

LinearSVC: Support Vector Machine with linear kernel

# Performance Metrics
Each model is evaluated using:

Accuracy

Precision

Recall

F1-Score

# Usage
Load the IMDB Dataset CSV file

Run the notebook cells sequentially

The models will be trained and evaluated automatically

Results are displayed with comparative analysis

# Results
The notebook provides a detailed comparison of model performance, helping identify the most effective approach for movie review sentiment analysis. Typically, LinearSVC and Logistic Regression tend to perform well on text classification tasks like this.


