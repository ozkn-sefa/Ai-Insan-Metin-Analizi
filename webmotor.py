import os
import json
import traceback
from typing import Dict, Tuple, Any, List

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# 0. Ayarlar ve Yollar
# ----------------------------
MODELS_DIR = "modeller"
SHARED_EMBEDDING_PATH = os.path.join(MODELS_DIR, "bert_st_model_gpu.joblib")

MODELS_CONFIG: Dict[str, Tuple[Any, str]] = {
    "1. MLP": (os.path.join(MODELS_DIR, "model1.pt"), "bert_gpu"),
    "2. LogReg": (os.path.join(MODELS_DIR, "model2.joblib"), "bert_skl_single"),
    "3. 1D CNN": (os.path.join(MODELS_DIR, "model3.pt"), "bert_gpu"),
    "4. BiLSTM": (os.path.join(MODELS_DIR, "model4.pt"), "bert_gpu"),
    "5. BERT-MLP": (os.path.join(MODELS_DIR, "bert_mlp_gpu.pt"), "bert_gpu"),
}

loaded_models: Dict[str, Any] = {}
shared_embedding_model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ----------------------------
# 1. PyTorch Model Sınıfları
# ----------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))


class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.fc_input_dim = 64 * (input_dim // 4)
        self.fc = nn.Linear(self.fc_input_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout_rate=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.unsqueeze(1)
        _, (hidden, _) = self.lstm(x)
        last_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        x = self.dropout(last_hidden)
        return torch.sigmoid(self.fc(x))


def get_model_class_by_path(path: str, input_dim: int):
    if "model3.pt" in path:
        return CNN1DClassifier(input_dim).to(device)
    elif "model4.pt" in path:
        return BiLSTMClassifier(input_dim).to(device)
    else:
        return MLPClassifier(input_dim).to(device)


# ----------------------------
# 2. Model Yükleme
# ----------------------------
def load_all_models():
    global loaded_models, shared_embedding_model
    success_count = 0

    print(f"[*] Sistem başlatılıyor. Cihaz: {device}")

    if os.path.exists(SHARED_EMBEDDING_PATH):
        try:
            shared_embedding_model = joblib.load(SHARED_EMBEDDING_PATH)
            print("[OK] Ortak Sentence Transformer yüklendi.")
        except Exception as e:
            print(f"[ERR] Embedding modeli yüklenirken hata: {e}")
            return
    else:
        print(f"[ERR] KRİTİK: Embedding dosyası bulunamadı: {SHARED_EMBEDDING_PATH}")
        return

    for name, (clf_path, model_type) in MODELS_CONFIG.items():
        if not os.path.exists(clf_path):
            print(f"[WARN] Dosya eksik, atlanıyor: {clf_path}")
            continue

        try:
            if model_type == "bert_gpu":
                input_dim = 768
                model_clf = get_model_class_by_path(clf_path, input_dim)
                model_clf.load_state_dict(torch.load(clf_path, map_location=device))
                model_clf.eval()
                loaded_models[name] = (model_clf, model_type)
                success_count += 1
            elif model_type == "bert_skl_single":
                clf = joblib.load(clf_path)
                loaded_models[name] = (clf, model_type)
                success_count += 1
            print(f"[OK] {name} başarıyla yüklendi.")
        except Exception as e:
            print(f"[ERR] {name} yüklenemedi: {e}")
            # traceback.print_exc() # Detaylı hata için açılabilir

    print(f"\n[SUMMARY] {success_count}/{len(MODELS_CONFIG)} model aktif.")


# ----------------------------
# 3. Analiz Mantığı
# ----------------------------
def get_single_model_prediction(model_obj, model_type, text_embedding):
    try:
        if model_type == "bert_skl_single":
            # Sklearn modelleri (Random Forest/LogReg) için veri tipi dönüşümü
            inp = np.asarray(text_embedding).astype(np.float64).reshape(1, -1)
            prob = model_obj.predict_proba(inp)
            return float(prob[0][1])
        elif model_type == "bert_gpu":
            emb_tensor = torch.tensor(text_embedding, dtype=torch.float32).to(device)
            if emb_tensor.ndim == 1:
                emb_tensor = emb_tensor.unsqueeze(0)
            with torch.no_grad():
                pred = model_obj(emb_tensor)
            return float(pred.cpu().numpy()[0][0])
    except Exception as e:
        print(f"Tahmin Hatası: {e}")
        return 0.5  # Hata durumunda nötr dön
    return 0.0


def analyze_text_multi_server(text: str) -> Dict[str, Any]:
    if not shared_embedding_model or not loaded_models:
        raise RuntimeError("Modeller hazır değil.")

    emb = shared_embedding_model.encode([text], device=device)
    results = {}
    ai_probs = []
    ai_votes = 0
    human_votes = 0
    max_conviction = -1.0
    best_model = "N/A"
    final_label = "N/A"

    for name, (model_obj, m_type) in loaded_models.items():
        prob = get_single_model_prediction(model_obj, m_type, emb)
        results[name] = prob
        ai_probs.append(prob)

        if prob >= 0.5:
            ai_votes += 1
        else:
            human_votes += 1

        conviction = prob if prob >= 0.5 else (1.0 - prob)
        if conviction > max_conviction:
            max_conviction = conviction
            best_model = name
            final_label = "AI" if prob >= 0.5 else "İnsan"

    return {
        "results": results,
        "avg_ai_prob": float(np.mean(ai_probs)) if ai_probs else 0,
        "std_dev_ai_prob": float(np.std(ai_probs)) if ai_probs else 0,
        "ai_votes": ai_votes,
        "human_votes": human_votes,
        "best_model_name": best_model,
        "final_decision_label": final_label,
        "final_percent": max_conviction * 100
    }



# 4. Flask Rotaları

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "status": "ok",
        "device": device,
        "loaded_models_count": len(loaded_models),
        "models": list(loaded_models.keys())
    }), 200


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Metin bulunamadı"}), 400

        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Metin boş olamaz"}), 400

        return jsonify(analyze_text_multi_server(text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_all_models()
    app.run(host="0.0.0.0", port=5000, debug=False)