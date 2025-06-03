import os
import sys
import time
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.curdir))

from models.lstm.lstm import LSTMClassifier
from models.lstm.lstm_attn import AttentionModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(train_loader, test_loader, vocab, embedding_weights, output_size,
                use_attention=False, batch_size=64, embedding_dim=100,
                hidden_size=128, dropout=0.5, epochs=100, patience=10,
                save_dir='./models/checkpoints', dataset_name='dataset', model_name='model'):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}_{model_name}.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = AttentionModel if use_attention else LSTMClassifier
    model = model_cls(batch_size, output_size, hidden_size, len(vocab),
                      embedding_dim, embedding_weights, dropout)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = 0
    patience_counter = 0

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, batch_size=inputs.size(0))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = correct / total
        
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs, batch_size=inputs.size(0))
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved: {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    end_time = time.time()
    training_time = end_time - start_time

    # Load best model before inference time measurement
    model.load_state_dict(torch.load(save_path))
    model.eval()
    dummy_input, _ = next(iter(test_loader))
    dummy_input = dummy_input.to(device)
    with torch.no_grad():
        inf_start = time.time()
        _ = model(dummy_input, batch_size=dummy_input.size(0))
        inf_end = time.time()
    avg_infer_time = (inf_end - inf_start) / dummy_input.size(0)

    num_params = count_parameters(model)
    return {
        "params": num_params,
        "train_time": training_time,
        "val_acc": best_val_acc,
        "infer_time": avg_infer_time,
        "model": model,
        "model_path": save_path
    }

if __name__ == '__main__':
    from data.getdl import prepare_imdb_dataloaders, prepare_sms_dataloaders, prepare_newsgroups_dataloaders
    glove_path = './data/glove.6B.100d.txt'

    datasets = {
        "IMDB": prepare_imdb_dataloaders('./data/IMDB Dataset.csv', glove_path),
        "SMS": prepare_sms_dataloaders('./data/spam.csv', glove_path),
        "20Newsgroups": prepare_newsgroups_dataloaders(glove_path)
    }
    
    results = []
    for ds_name, data in datasets.items():
        if ds_name == "20Newsgroups":
            train_loader, test_loader, vocab, emb, num_classes = data
            output_size = num_classes
        else:
            train_loader, test_loader, vocab, emb = data
            output_size = 2
        
        for use_attn in [False, True]:
            model_type = "LSTM+Attn" if use_attn else "LSTM"
            print(f"\nTraining {model_type} on {ds_name} dataset...")
            res = train_model(train_loader, test_loader, vocab, emb, output_size,
                              use_attention=use_attn, epochs=50, patience=5,
                              dataset_name=ds_name, model_name=model_type.replace('+',''))
            results.append({
                "Dataset": ds_name,
                "Model": model_type,
                "Params": res["params"],
                "Train Time (s)": f"{res['train_time']:.2f}",
                "Val Accuracy": f"{res['val_acc']:.4f}",
                "Infer Time/Sample (s)": f"{res['infer_time']:.6f}",
                "Model Path": res["model_path"]
            })

    # Print summary table
    print("\n===== Summary =====")
    print(f"{'Dataset':<12} {'Model':<12} {'Params':<12} {'Train Time(s)':<15} {'Val Acc':<10} {'Infer Time/Sample':<18} {'Model Path'}")
    print("-" * 100)
    for r in results:
        print(f"{r['Dataset']:<12} {r['Model']:<12} {r['Params']:<12,} {r['Train Time (s)']:<15} {r['Val Accuracy']:<10} {r['Infer Time/Sample (s)']:<18} {r['Model Path']}")

    df_results = pd.DataFrame(results)
    df_results.to_csv("./compare_lstm.csv", index=False)