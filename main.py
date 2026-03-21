import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import os

from tokenizer_utils import SimpleTokenizer, pad_sequences
from dataset import load_dataset


class ComplaintsDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DoctorNet(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        pooled = lstm_out.mean(dim=1)
        out = self.fc(pooled)
        return out


def train_and_save_model():
    print("Загрузка датасета...")
    df = load_dataset()
    complaints = df["complaint"].tolist()

    specialists = [
        "ЛОР",
        "Пульмонолог",
        "Кардиолог",
        "Гастроэнтеролог",
        "Невролог",
        "Хирург",
        "Стоматолог",
        "Дерматолог",
        "Окулист"
    ]

    doctor_to_label = {doctor: idx for idx, doctor in enumerate(specialists)}
    labels = [doctor_to_label[doctor] for doctor in df["doctor"]]

    print(f"Загружено {len(complaints)} жалоб")
    print("Распределение по врачам:")
    print(df["doctor"].value_counts())
    print()

    tokenizer = SimpleTokenizer()
    tokenizer.fit_on_texts(complaints)
    print(f"Создан словарь из {len(tokenizer.word_index)} слов")

    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("Токенизатор сохранен в tokenizer.pkl")

    sequences = tokenizer.texts_to_sequences(complaints)
    maxlen = 50
    X = pad_sequences(sequences, maxlen)
    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)

    dataset = ComplaintsDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DoctorNet(len(tokenizer.word_index), num_classes=len(specialists))
    print(f"Создана модель с размером словаря: {len(tokenizer.word_index)}")
    print("Количество специалистов:", len(specialists))
    print("Максимальная метка:", max(labels))
    print("Размер выходного слоя модели:", model.fc.out_features)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nНачало обучения...")
    for epoch in range(10):
        total_loss = 0
        model.train()

        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "doctor_model.pth")
    print("Модель сохранена в doctor_model.pth")

    metadata = {
        "maxlen": maxlen,
        "specialists": specialists,
        "vocab_size": len(tokenizer.word_index),
        "word_index": tokenizer.word_index
    }
    with open("model_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print("Метаданные сохранены в model_metadata.pkl")

    model.eval()
    print("\nТестирование модели на примерах:")

    test_complaints = [
        "боль в голове",
        "кашель и температура",
        "болит живот и тошнота",
        "болит ухо горло нос",
        "болит живот после операции",
        "частые головные боли",
        "болит зуб",
        "на коже сыпь",
        "гнойные выделения из глаза"
    ]

    for complaint in test_complaints:
        seq = tokenizer.texts_to_sequences([complaint])
        padded = pad_sequences(seq, maxlen)
        padded = torch.tensor(padded, dtype=torch.long)

        with torch.no_grad():
            logits = model(padded)
            probs = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()

        doctor = specialists[predicted_class]
        print(f"Жалоба: '{complaint}' -> Врач: {doctor}")

    print("\n" + "=" * 50)
    print("Модель успешно обучена и сохранена!")
    print("Теперь можно запускать веб-приложение.")
    print("=" * 50)

    return model, tokenizer, maxlen, specialists


if __name__ == "__main__":
    print("=" * 50)
    print("Обучение модели для веб-приложения")
    print("=" * 50)

    files_needed = ["tokenizer.pkl", "doctor_model.pth", "model_metadata.pkl"]
    files_exist = all(os.path.exists(f) for f in files_needed)

    if files_exist:
        print("Обнаружены сохраненные файлы модели.")
        choice = input("Хотите заново обучить модель? (y/n): ")
        if choice.lower() == "y":
            train_and_save_model()
        else:
            print("Используем существующую модель.")
    else:
        print("Сохраненные файлы не найдены. Начинаем обучение...")
        train_and_save_model()