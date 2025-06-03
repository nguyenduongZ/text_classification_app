import os, sys
sys.path.append(os.path.abspath(os.curdir))
import pickle
import time
import csv
from flask import Flask, render_template, request, jsonify, send_file
from utils.preprocessing import preprocess_text
from transformers import pipeline as hf_pipeline
from inference import *
import matplotlib.pyplot as plt
import uuid

from data.getdl import download_all_models
download_all_models()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html', tasks=tasks)

@app.route('/models', methods=['POST'])
def get_models():
    data = request.get_json()
    task = data.get('task')
    models = tasks.get(task, [])
    return jsonify(models)

@app.route('/predict', methods=['POST'])
def predict():
    task = request.form.get('task')
    model_name = request.form.get('model')
    if not task or not model_name:
        return jsonify(error='Task and model must be selected.'), 400

    text_inputs = []
    raw_lines = []
    if 'text_input' in request.form and request.form['text_input'].strip():
        raw_text = request.form['text_input']
        processed_text = preprocess_text(raw_text)
        if len(processed_text.split()) < 3:
            return jsonify(error='Input text must be at least 3 words after preprocessing.'), 400
        text_inputs.append(processed_text)
        raw_lines.append(raw_text)
    elif 'file' in request.files:
        uploaded_file = request.files['file']
        try:
            content = uploaded_file.read().decode('utf-8')
        except UnicodeDecodeError:
            uploaded_file.stream.seek(0)
            content = uploaded_file.read().decode('latin1')
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        for line in lines:
            processed_line = preprocess_text(line)
            if len(processed_line.split()) >= 5:
                text_inputs.append(processed_line)
                raw_lines.append(line)
        if not text_inputs:
            return jsonify(error='No valid lines with at least 5 words found in file after preprocessing.'), 400
    else:
        return jsonify(error='Please input text or upload a file.'), 400

    try:
        if model_name in ML_MODEL_PATHS.get(task, {}):
            results = predict_ml(text_inputs, task, model_name)
        elif model_name in LSTM_MODEL_PATHS.get(task, {}):
            results = predict_lstm(text_inputs, task, model_name)
        elif model_name in TRANSFORMER_MODELS.get(task, {}):
            results = predict_transformer(text_inputs, task, model_name)
        else:
            return jsonify(error=f"Model '{model_name}' not supported for task '{task}'."), 400
    except Exception as e:
        return jsonify(error=str(e)), 500

    # Generate CSV output
    result_id = uuid.uuid4().hex
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], f'results_{result_id}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Text', 'Predicted Label', 'Confidence'])
        for orig_text, res in zip(raw_lines, results):
            writer.writerow([orig_text, res['label'], f"{res['confidence']*100:.2f}%"])

    # Plot chart
    confidences = [res['confidence']*100 for res in results]
    labels = [f"{res['label']}" for res in results]
    fig, ax = plt.subplots(figsize=(8, max(3, len(results)*0.5)))
    bars = ax.barh(range(len(results)), confidences, color='skyblue')
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Confidence Scores')
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{confidences[i]:.1f}%", va='center')
    plt.tight_layout()
    chart_path = os.path.join(app.config['UPLOAD_FOLDER'], f'chart_{result_id}.png')
    plt.savefig(chart_path)
    plt.close()

    return jsonify({
        'results': results,
        'csv_url': f"/download/{os.path.basename(csv_path)}",
        'chart_url': f"/{chart_path}"
    })

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
