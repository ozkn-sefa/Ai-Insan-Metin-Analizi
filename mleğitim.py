# Gerekli Kütüphaneler
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib
import os
import time
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# ======= Ayarlar =======
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBEDDING_MODEL = 'all-mpnet-base-v2'
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.3
SAMPLE_COUNT = 2990  # Her sınıftan alınacak örnek sayısı
MODELS_DIR = "modeller"

# Modeller klasörü yoksa oluştur
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# ======= Veri Yükleme, Filtreleme ve Dengeleme =======
DATA_FILE = os.path.join("dökümanlar", "PROJE_VERI_SETI_SON_HALI.xlsx")

if not os.path.exists(DATA_FILE):
    print(f"🚨 Hata: '{DATA_FILE}' dosyası bulunamadı.")
    exit()

try:
    df_full = pd.read_excel(DATA_FILE)
    df_full['kimden'] = df_full['kimden'].astype(str).str.lower()

    ai_df = df_full[df_full['kimden'] == 'ai'].copy()
    insan_df = df_full[df_full['kimden'] == 'insan'].copy()

    min_count = min(len(ai_df), len(insan_df), SAMPLE_COUNT)

    ai_sampled = ai_df.sample(min_count, random_state=42)
    insan_sampled = insan_df.sample(min_count, random_state=42)

    df_balanced = pd.concat([ai_sampled, insan_sampled], ignore_index=True)
    df_balanced["label"] = (df_balanced["kimden"] == "ai").astype(int)

    X = df_balanced["metin"].astype(str).tolist()
    y = df_balanced["label"].tolist()

    print(f"✅ Veri Seti Hazır: Toplam {len(df_balanced)} örnek.")

except Exception as e:
    print(f"🚨 Veri hatası: {e}")
    exit()

# ======= 📦 BERT Embedding Oluşturma =======
print("📦 BERT embedding (Sentence Transformer) yükleniyor ve oluşturuluyor...")
model_st = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
X_emb_np = model_st.encode(X, batch_size=BATCH_SIZE, show_progress_bar=True, device=DEVICE)
X_emb = torch.tensor(X_emb_np, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# ======= 🛠️ Veri Hazırlama =======
dataset = TensorDataset(X_emb, y_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

X_train_np = X_emb[train_ds.indices].cpu().numpy()
y_train_np = y_tensor[train_ds.indices].cpu().numpy().ravel()
X_test_np = X_emb[test_ds.indices].cpu().numpy()
y_test_np = y_tensor[test_ds.indices].cpu().numpy().ravel()

INPUT_DIM = X_emb.shape[1]
results = []
model_counter = 1


# ======= Model Mimarileri =======
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, dropout_rate=DROPOUT_RATE):
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
    def __init__(self, input_dim, dropout_rate=DROPOUT_RATE):
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
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True,
                            dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.unsqueeze(1)
        _, (hidden, _) = self.lstm(x)
        last_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        x = self.dropout(last_hidden)
        return torch.sigmoid(self.fc(x))


# ======= Eğitim Fonksiyonları =======
def train_and_evaluate_nn(model, model_name, train_loader, test_loader):
    global model_counter
    print(f"\n--- {model_counter}. Eğitiliyor: {model_name} ---")
    start_time = time.time()
    model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad();
            preds = model(xb);
            loss = criterion(preds, yb);
            loss.backward();
            optimizer.step()

    training_time = time.time() - start_time
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in test_loader:
            p = model(xb.to(DEVICE))
            all_probs.extend(p.cpu().numpy().flatten())
            all_labels.extend(yb.numpy().flatten())

    all_preds = [1 if p > 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, all_preds)

    # Kayıt
    save_path = os.path.join(MODELS_DIR, f"model{model_counter}.pt")
    torch.save(model.state_dict(), save_path)
    results.append({'model_name': model_name, 'accuracy': acc, 'time': training_time})
    print(f"✅ Kaydedildi: {save_path}")
    model_counter += 1


def train_and_evaluate_ml(model, model_name, X_train, y_train, X_test, y_test):
    global model_counter
    print(f"\n--- {model_counter}. Eğitiliyor: {model_name} ---")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    acc = accuracy_score(y_test, model.predict(X_test))

    save_path = os.path.join(MODELS_DIR, f"model{model_counter}.joblib")
    joblib.dump(model, save_path)
    results.append({'model_name': model_name, 'accuracy': acc, 'time': training_time})
    print(f"✅ Kaydedildi: {save_path}")
    model_counter += 1


# ======= Modelleri Çalıştır =======
train_and_evaluate_nn(MLPClassifier(INPUT_DIM), "BERT_MLP", train_loader, test_loader)
train_and_evaluate_ml(LogisticRegression(max_iter=1000), "BERT_LogReg", X_train_np, y_train_np, X_test_np, y_test_np)
train_and_evaluate_nn(CNN1DClassifier(INPUT_DIM), "BERT_CNN", train_loader, test_loader)
train_and_evaluate_nn(BiLSTMClassifier(INPUT_DIM), "BERT_BiLSTM", train_loader, test_loader)

# ======= 📦 KRİTİK: Ortak Dosyaları Kaydet =======
print("\n📦 Ortak sistem dosyaları kaydediliyor...")

# 1. BERT Embedding Modelini Kaydet (Flask için en önemlisi)
emb_save_path = os.path.join(MODELS_DIR, "bert_st_model_gpu.joblib")
joblib.dump(model_st, emb_save_path)

# 2. JSON Konfigürasyonu
config = {
    "embedding_model": EMBEDDING_MODEL,
    "input_dim": INPUT_DIM,
    "device": DEVICE,
    "models": {
        "1. MLP": "model1.pt", "2. LogReg": "model2.joblib",
        "3. 1D CNN": "model3.pt", "4. BiLSTM": "model4.pt"
    }
}
with open(os.path.join(MODELS_DIR, "model_config.json"), "w") as f:
    json.dump(config, f, indent=4)

# 3. Excel Raporu
pd.DataFrame(results).to_excel("model_karsilastirma_sonuclari.xlsx", index=False)

print("\n" + "=" * 50)
print(f"🚀 TAMAMLANDI! Tüm modeller '{MODELS_DIR}' klasörüne kaydedildi.")
print("=" * 50)