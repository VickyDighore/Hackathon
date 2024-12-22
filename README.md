# Hackathon

DATA SCIENCE HACKATHON

E-Commerce Product Categorisation

Problem Statement: In the rapidly evolving world of eCommerce, accurate product categorization is crucial for ensuring seamless customer experiences, reducing search friction, and increasing product discoverability. However, the sheer volume of diverse products poses a significant challenge. Current classification systems struggle to handle ambiguities, unconventional naming conventions, and multi-language data. This hackathon aims to address these challenges by inviting participants to create innovative solutions that enhance product categorization efficiency, accuracy, and scalability. Develop a text classification model that categorizes products with maximum accuracy based on description of the product.

Problem-Solving Approach for E-Commerce Product Categorization:

Understanding the Problem Define the problem and objectives: Categorize products accurately based on their descriptions to enhance customer experience and improve search efficiency. Identify challenges like handling ambiguous product names, multi-language data, and unconventional naming. Data Exploration and Preparation Explore the dataset to understand its structure, features, and distributions. Identify missing values, outliers, and inconsistencies in the data. Clean and preprocess the text data by: Removing punctuation, stop words, and special characters. Handling missing values (e.g., filling with "unknown" or dropping). Standardizing text (e.g., lowercasing). Descriptive Analysis Perform descriptive statistics to understand key insights. Visualize category distributions using bar plots to detect imbalances. Use word clouds to identify prominent terms in product descriptions. Feature Engineering Transform text data into numerical features: Use techniques like TF-IDF, word embeddings (Word2Vec, GloVe), or advanced models (BERT). Experiment with n-grams and adjust feature extraction parameters for better performance. Predictive Modeling Choose a suitable machine learning algorithm (e.g., Random Forest, Gradient Boosting, or deep learning models like LSTMs). Split the data into training and testing sets. Train an initial model to establish a baseline accuracy. Fine-Tuning Optimize the model using hyperparameter tuning techniques (e.g., GridSearchCV). Experiment with different architectures, parameter settings, and feature subsets. Validate using cross-validation to ensure generalizability. Model Evaluation Evaluate the model on metrics like accuracy, precision, recall, and F1-score. Use confusion matrices to identify misclassified categories. Address performance gaps by analyzing patterns in misclassifications. Enhancements Address domain-specific challenges (e.g., ambiguous product names) by integrating domain knowledge. Use ensemble methods or hybrid models to improve accuracy. Incorporate multilingual support by leveraging language-specific embeddings or translation preprocessing. Visualization Present key insights through clear and concise visualizations. Display feature importance, misclassification patterns, and distribution insights. Documentation and Delivery Provide a well-documented Jupyter Notebook covering the end-to-end process. Ensure modularity and clarity in the code for reproducibility. Create a video walkthrough explaining: The problem and approach. Data analysis and insights. Model performance and business value. Scalability Design the solution for real-time categorization and scalability. Optimize the pipeline for deployment in production environments.

Codebase Structure
Notebooks: This is the folder containing all the Jupyter Notebooks that have been used for Exploratory Data Analysis, training and testing of the Machine Learning and Deep Learning models.

requirements.txt: This file contains all the dependencies of the project that are needed to reproduce the development environment

Dataset: This folder contains all the datasets (imbalanced and balanced) in CSV format.

The following tasks were undertaken for the Multiclass Classification of e-commerce products based on their description:

The dataset and several of its hidden parameters were visualised (using libraries like seaborn, matplotlib, yellowbrick, etc). This then helped in data cleaning as several words from the Word Cloud were removed from the corpus as they did not contribute much in terms of Product Classification.
It was decided to move forward by only using the root of the Product Category Tree as the Primary label/category for classification.
Data cleaning, preprocessing and resampling was then performed to balance out the given dataset.
After a detailed analysis of the dataset through visualisation and other parameters, it was decided to categorise the products in the following 13 categories and remove the noise (other miscellaneous categories having less than 10 products):
a) Clothing
b) Jewellery
c) Sports & Fitness
d) Electronics
e) Babycare
f) Home Furnishing & Kitchen
g) Personal Accessories
h) Automotive
i) Pet Supplies
j) Tools & Hardware
k) Ebooks
l) Toys & School Supplies
m) Footwear
Then, the following Machine Learning algorithms (using scikit-learn libraries) were applied on the dataset:
a) Logistic Regression (Binary and Multiclass variants)
b) Linear Support Vector Machine
c) Multinomial Naive Bayes
d) Decision Tree
d) Random Forest Classifier
e) K Nearest Neighbours
Even though good accuracy was achieved using the ML models, the following Deep Learning Models (using PyTorch framework) were also implemented on the dataset:
1. Transformer based models like:
a) Bidirectional Encoder Representations from Transformers (BERT)
b) RoBERTa
c) DistilBERT
d) XLNet
2. Recurrent Neural Network based Long-Short Term Memory(LSTM)
STEP 1: Exploratory Data Analysis and Data Preprocessing
An in depth analysis of the dataset was done with the help of Word Clouds, Bar Graphs, TSNE Visualizations, etc to get an idea about the most frequent unigrams in the Product Description, distribution of products and brands across the different Product Categories, analysis of the length of the description, etc.
For Data Cleaning, Contraction Mapping, removal of custom stopwords, URLs, Tokenization and Lemmatization was done.
Because of the clear imbalance in the dataset, balancing techniques like Oversampling and Undersampling were performed on the dataset as well. These were then saved in the form of a CSV file.
STEP 2: Machine Learning Models for Product Categorization
The above mentioned 6 ML algorithms were applied on the imbalanced, oversampling balanced and undersampling balanced datasets. Noise was removed from each of these datasets and these datasets had already been cleaned and preprocessed in the previous notebook.
Several evaluation metrics like Classification Report, Confusion Matix, Accuracy Score, ROC Curves and AUC Scores were used for the comparison of the models. The Validation score of the ML algorithms when applied on the dataset are tabulated below:
ML Algorithm	Validation Accuracy on Imbalanced Dataset	Validation Accuracy on Balanced Dataset (Oversampling)	Validation Accuracy on Balanced Dataset (Undersampling)
Logistic Regression (Binary)	0.9654	0.9756	0.9486
Logistic Regression (Multiclass)	0.9735	0.9893	0.9654
Naive Bayes	0.9096	0.9602	0.9054
Linear SVM	0.9799	0.9958	0.9749
Decision Trees	0.70170	0.6883	0.7561
Random Classifier	0.9209	0.9367	0.9235
K Nearest Neighbours	0.9564	0.98	0.9453
From the above table, we can clearly see that Linear Support Vector Machine algorithm performed the best across all the three datasets.
STEP 3: Deep Learning Models for Product Categorization
The Deep Learning Models were only trained and evaluated on the dataset that was baalnced using the Undersampling technique.
After a detailed study of all the Transformer based Deep Learning algorithms like BERT, RoBERTa, DistilBERT, XLNet and Recurrent Neural Network based LSTM, it was decided that BERT (uncased, base, with all the layers freezed except the last one) worked the best on our dataset by giving an f1-score of 0.98.
Confusion Matrix

Future Work
Feature extraction can be performed on the Product Category Tree column in order to find a more detailed class to which a product can belong.
Using other advanced data balancing techniques like SMOTE, etc.
Training and evaluating the Deep Learning model on datasets other than the undersampled one. These models could then be tested on a variety of e-commerce data available online to understand the scalability of the model when it comes to dealing with real-world data.
Using Named Entity Recognition techniques to figure out brands that make products belonging to a specific category.
References
Vasvani et. al. Attention is all you need. Nips 2017
Simple Transformers
Tranformer Models by HuggingFace
Multiclass text classification with Deep Learning
Multiclass text classification using LSTM
