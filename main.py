import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
from dataset import load_dataset
import os

#Специалисты
specialists = [
    "ЛОР", "Пульмонолог", "Кардиолог", "Гастроэнтеролог",
    "Невролог", "Хирург", "Стоматолог", "Дерматолог", "Окулист"
]
label2id = {label: idx for idx, label in enumerate(specialists)}
id2label = {idx: label for label, idx in label2id.items()}

def train_and_save_model():
    print("Загрузка датасета...")
    df = load_dataset()
    df = df[df["doctor"].isin(specialists)].copy()  # на всякий случай
    df["label"] = df["doctor"].map(label2id)

    print(f"Загружено {len(df)} жалоб")
    print(df["doctor"].value_counts())

    #Загружаем медицинскую RuBioRoBERTa
    model_name = "alexyalunin/RuBioRoBERTa"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(specialists),
        id2label=id2label,
        label2id=label2id
    )

    #Подготовка датасета
    dataset = Dataset.from_pandas(df[["complaint", "label"]])
    def tokenize_function(examples):
        return tokenizer(examples["complaint"], truncation=True, max_length=128, padding="max_length")
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    #Обучение
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        logging_dir="./logs",
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(-1)
        return {"accuracy": (predictions == labels).mean()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],      # ← вот эта строчка должна быть!
        compute_metrics=compute_metrics,
    )

    print("Начинаем обучение (это может занять пару минут)")
    trainer.train()

    #Сохраняем
    model.save_pretrained("doctor_model")
    tokenizer.save_pretrained("tokenizer")
    print("Модель и токенайзер сохранены в doctor_model/ и tokenizer/")

    #Тест на примерах
    print("\nТестируем модельку:")
    test_complaints = [
        "боль в голове", "кашель и температура", "болит живот и тошнота",
        "болит ухо горло нос", "болит колено", "болит зуб", "на коже сыпь"
    ]
    for text in test_complaints:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs.logits.argmax(-1).item()
        print(f"Жалоба: '{text}' → {id2label[pred]}")

    print("\nВсё готово. Теперь можно запускать веб-приложение")

if __name__ == "__main__":
    if os.path.exists("doctor_model") and os.path.exists("tokenizer"):
        choice = input("Модель уже есть. Переобучить? (y/n): ")
        if choice.lower() != "y":
            print("Используем существующую модель.")
            exit()
    train_and_save_model()
