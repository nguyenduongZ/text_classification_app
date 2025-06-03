import os
import time
import pickle
import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import warnings
warnings.filterwarnings("ignore")

MODELS = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')),
    ('Naive Bayes', MultinomialNB(alpha=0.1)),
    ('SVM', LinearSVC(C=1.0, max_iter=1000)),
    ('Random Forest', RandomForestClassifier(n_estimators=200, max_depth=30, n_jobs=1)),
    ('KNN', KNeighborsClassifier(n_neighbors=5, weights='distance')),
    ('Decision Tree', tree.DecisionTreeClassifier())
]

results = []

def benchmark_models(X_train, X_test, y_train, y_test, dataset_name):
    best_acc = 0
    best_model_pipeline = None
    best_model_name = None
    
    for name, model in MODELS:
        pipeline = Pipeline(
            [
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')),
                ("clf", model)
            ]
        )
        
        start_train = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_train

        start_pred = time.time()
        y_pred = pipeline.predict(X_test)
        pred_time = time.time() - start_pred

        acc = accuracy_score(y_test, y_pred)
        
        results.append({
            'Dataset': dataset_name,
            'Model': name,
            'Accuracy': round(acc * 100, 2),
            'Train Time (s)': round(train_time, 2),
            'Predict Time (s)': round(pred_time, 4)
        })
        
        if acc > best_acc:
            best_acc = acc
            best_model_pipeline = pipeline
            best_model_name = name
    
    if best_model_pipeline is not None:
        os.makedirs('./models/checkpoints', exist_ok=True)
        filename = f"{dataset_name.replace(' ', '_')}_{best_model_name.replace(' ', '_')}.pkl"
        filename_path = os.path.join('./models/checkpoints', filename)
        with open(filename_path, 'wb') as f:
            pickle.dump(best_model_pipeline, f)
        print(f"Save best model for {dataset_name}: {filename_path} (Accuracy: {round(best_acc, 2)}%)")     

# IMDB Dataset
print('IMDB Dataset')
imdb_df = pd.read_csv('./data/IMDB Dataset.csv')
X_train, X_test, y_train, y_test = train_test_split(
    imdb_df['review'], 
    imdb_df['sentiment'], 
    test_size=0.2, 
    random_state=42
)
benchmark_models(X_train, X_test, y_train, y_test, "IMDB")

# SMS Spam Dataset
print('SMS Spam Dataset')
sms_df = pd.read_csv('./data/spam.csv', encoding='latin1')
sms_df['label'] = sms_df['v1'].map({'ham': 0, 'spam': 1})
X_train, X_test, y_train, y_test = train_test_split(
    sms_df['v2'], 
    sms_df['label'], 
    test_size=0.2, 
    random_state=42
)
benchmark_models(X_train, X_test, y_train, y_test, "SMS Spam")

# 20 Newsgroups Dataset
print('20 Newsgroups Dataset')
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X_train, X_test, y_train, y_test = train_test_split(
    newsgroups.data,
    newsgroups.target,
    test_size=0.2, 
    random_state=42
)
benchmark_models(X_train, X_test, y_train, y_test, "20 Newsgroups")

df_results = pd.DataFrame(results)
print(f"\nBenchmark Results:")
print(df_results.sort_values(by=['Dataset', 'Accuracy'], ascending=[True, False]))

df_results.to_csv("./compare_ml.csv", index=False)