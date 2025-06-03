import os
import torch
import pickle

from transformers import pipeline as hf_pipeline

from models.lstm.lstm import LSTMClassifier
from models.lstm.lstm_attn import AttentionModel
from utils.glove_embeddings import load_glove_embeddings
from data.getdl import load_vocab_from_pickle 
from utils.preprocessing import *

loaded_ml_models = {}
loaded_lstm_models = {}
loaded_transformer_models = {}

TOPIC_LABEL_MAP = {
    0: 'alt.atheism',
    1: 'comp.graphics',
    2: 'comp.os.ms-windows.misc',
    3: 'comp.sys.ibm.pc.hardware',
    4: 'comp.sys.mac.hardware',
    5: 'comp.windows.x',
    6: 'misc.forsale',
    7: 'rec.autos',
    8: 'rec.motorcycles',
    9: 'rec.sport.baseball',
    10: 'rec.sport.hockey',
    11: 'sci.crypt',
    12: 'sci.electronics',
    13: 'sci.med',
    14: 'sci.space',
    15: 'soc.religion.christian',
    16: 'talk.politics.guns',
    17: 'talk.politics.mideast',
    18: 'talk.politics.misc',
    19: 'talk.religion.misc'
}

ML_MODEL_PATHS = {
    "sentiment_analysis": {
        "Logistic_Regression": "./models/checkpoints/IMDB_Logistic_Regression.pkl"
    },
    "spam_detection": {
        "Naive_Bayes": "./models/checkpoints/SMS_Spam_Naive_Bayes.pkl"
    },
    "topic_classification": {
        "Naive_Bayes": "./models/checkpoints/20_Newsgroups_Naive_Bayes.pkl"
    }
}

LSTM_MODEL_PATHS = {
    'spam_detection': {
        'LSTM': "./models/checkpoints/SMS_LSTM.pt",
        'LSTMAttn': "./models/checkpoints/SMS_LSTMAttn.pt"
    }
}

VOCAB_PATHS = {
    'spam_detection': './models/checkpoints/spam_detection_vocab.pkl'
}

GLOVE_PATH = './data/glove.6B.100d.txt'

TRANSFORMER_MODELS = {
    'sentiment_analysis': {
        'distilbert': {
            'model': 'distilbert-base-uncased-finetuned-sst-2-english',
            'label_map': {'POSITIVE': 'positive', 'NEGATIVE': 'negative'}
        },
        'bert-base': {
            'model': 'textattack/bert-base-uncased-SST-2',
            'label_map': {'POSITIVE': 'positive', 'NEGATIVE': 'negative'}
        }
    },
    'spam_detection': {
        'bert-spam': {
            'model': 'mrm8488/bert-tiny-finetuned-sms-spam-detection',
            'label_map': {'LABEL_0': 'ham', 'LABEL_1': 'spam'}
        }
            
    },
    'topic_classification': {
        'bert-topic': 'facebook/bart-large-mnli'
    }
}

tasks = {
    'sentiment_analysis': ['Logistic_Regression', 'distilbert', 'bert-base'],  
    'spam_detection': ['Naive_Bayes', 'LSTM', 'LSTMAttn', 'bert-spam'], 
    'topic_classification': ['Naive_Bayes', 'bert-topic'] 
}

def load_ml_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict_ml(texts, task, model_key):
    if model_key not in loaded_ml_models:
        model_path = ML_MODEL_PATHS.get(task, {}).get(model_key)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"ML model '{model_key}' for task '{task}' not found")
        loaded_ml_models[model_key] = load_ml_model(model_path)

    pipeline = loaded_ml_models[model_key]
    preds = pipeline.predict(texts)

    try:
        proba = pipeline.predict_proba(texts)
        confidences = proba.max(axis=1)
    except Exception:
        confidences = [1.0] * len(preds)

    if task == 'spam_detection':
        label_map = {0: 'ham', 1: 'spam'}
    elif task == 'topic_classification':
        label_map = TOPIC_LABEL_MAP
    else:
        label_map = {}

    return [
        {'text': text, 'label': label_map.get(pred, str(pred)), 'confidence': float(conf)}
        for text, pred, conf in zip(texts, preds, confidences)
    ]

def load_transformer_model(task, model_key):
    config = TRANSFORMER_MODELS.get(task, {}).get(model_key)
    if config is None:
        raise ValueError(f"Transformer model '{model_key}' for task '{task}' not found")

    model_name = config['model'] if isinstance(config, dict) else config
    label_map = config.get('label_map', {}) if isinstance(config, dict) else {}

    task_type = 'zero-shot-classification' if task == 'topic_classification' else 'sentiment-analysis'

    try:
        pipe = hf_pipeline(task_type, model=model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load transformer model '{model_name}': {e}")

    return pipe, label_map

def predict_transformer(texts, task, model_key):
    cache_key = f"{task}_{model_key}"
    if cache_key not in loaded_transformer_models:
        pipe, label_map = load_transformer_model(task, model_key)
        loaded_transformer_models[cache_key] = (pipe, label_map)
    else:
        pipe, label_map = loaded_transformer_models[cache_key]

    processed_texts = preprocess_corpus_for_transformer(texts)
    
    results = []
    for text, processed in zip(texts, processed_texts):
        try:
            if task == 'topic_classification':
                candidate_labels = [
                    'atheism discussion', 'computer graphics', 'ms windows issues', 'ibm pc hardware',
                    'mac hardware', 'windows x software', 'items for sale', 'automobiles',
                    'motorcycles', 'baseball sports', 'hockey sports', 'cryptography',
                    'electronics', 'medicine', 'space exploration', 'christian religion',
                    'gun politics', 'mideast politics', 'miscellaneous politics', 'religion miscellaneous'
                ]
                output = pipe(processed, candidate_labels, multi_label=False, truncation=True, max_length=512)
                label = output['labels'][0]
                confidence = output['scores'][0]
            else:
                output = pipe(processed, truncation=True, max_length=512)[0]
                label = output['label']
                confidence = output['score']
            mapped_label = label_map.get(label, label)
            results.append({'text': text, 'label': mapped_label, 'confidence': float(confidence)})
        except Exception as e:
            print(f"Error predicting '{text}': {e}")
            results.append({'text': text, 'label': 'unknown', 'confidence': 0.0})
    return results

def load_lstm_model(task, model_key):
    try:
        vocab_path = VOCAB_PATHS[task]
        model_path = LSTM_MODEL_PATHS[task][model_key]
    except KeyError:
        raise ValueError(f"LSTM model '{model_key}' not found for task '{task}'")

    vocab = load_vocab_from_pickle(vocab_path)
    weights = load_glove_embeddings(GLOVE_PATH, 100, vocab)

    model_cls = AttentionModel if "Attn" in model_key else LSTMClassifier
    model = model_cls(batch_size=64, output_size=2, hidden_size=128,
                      vocab_size=len(vocab), embedding_length=100,
                      weights=weights, dropout=0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval(), vocab

def predict_lstm(texts, task, model_key):
    cache_key = f"{task}_{model_key}"
    if cache_key not in loaded_lstm_models:
        model, vocab = load_lstm_model(task, model_key)
        loaded_lstm_models[cache_key] = (model, vocab)
    else:
        model, vocab = loaded_lstm_models[cache_key]

    device = next(model.parameters()).device
    results = []
    
    LABEL_MAPS = {
        'spam_detection': {0: 'ham', 1: 'spam'},
    }

    label_map = LABEL_MAPS.get(task, {})

    for text in texts:
        processed = preprocess_text_for_lstm(text, vocab)
        processed = torch.tensor(processed, dtype=torch.long)

        if processed.nelement() == 0:
            results.append({'text': text, 'label': 'unknown', 'confidence': 0.0})
            continue

        input_tensor = processed.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor, batch_size=1)
            probs = torch.softmax(output, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)

        pred_label = label_map.get(pred_class.item(), str(pred_class.item()))

        results.append({
            'text': text,
            'label': pred_label,
            'confidence': float(confidence.item())
        })

    return results
