import os
import torch
import numpy as np
import pickle
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from secure_excel_logger import SecureExcelLogger

app = FastAPI(title="Маршрутизация пациента")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

print("🤖 Загрузка модели...")

with open("model_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

maxlen = metadata["maxlen"]
specialists = metadata["specialists"]
vocab_size = metadata["vocab_size"]
word_index = metadata["word_index"]


class SimpleTokenizer:
    def __init__(self, word_index):
        self.word_index = word_index

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = [self.word_index.get(word, 0) for word in text.lower().split()]
            sequences.append(seq)
        return sequences


def pad_sequences(sequences, maxlen):
    padded = []
    for seq in sequences:
        if len(seq) > maxlen:
            seq = seq[:maxlen]
        else:
            seq = [0] * (maxlen - len(seq)) + seq
        padded.append(seq)
    return np.array(padded)


class DoctorNet(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_classes=8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size + 1, embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        pooled = lstm_out.mean(dim=1)
        out = self.fc(pooled)
        return out


tokenizer = SimpleTokenizer(word_index)

model = DoctorNet(vocab_size, num_classes=len(specialists))
model.load_state_dict(torch.load("doctor_model.pth", map_location=torch.device("cpu")))
model.eval()

logger = None
APP_MODE = os.getenv("APP_MODE", "local")

if APP_MODE == "local":
    try:
        logger = SecureExcelLogger(
            excel_path="cases.xlsx",
            key_path="secret.key"
        )
        print("✅ Защищённый логгер подключен")
    except Exception as e:
        print(f"⚠️ Ошибка инициализации логгера: {e}")
        logger = None
else:
    print("ℹ️ Demo mode: логирование отключено")

print("✅ Модель загружена")


def predict_complaint(text):
    text_lower = text.lower()

    dental_keywords = [
        "зуб", "зубы", "зубная", "зубной", "десн", "десна", "десны",
        "кариес", "пульпит", "пломб", "коронк", "мост", "протез",
        "челюст", "прикус", "брекет", "ортодонт", "имплант",
        "флюс", "абсцесс", "пародонт", "гингивит", "стоматолог"
    ]

    for keyword in dental_keywords:
        if keyword in text_lower:
            fake_probs = torch.zeros(1, len(specialists))
            dental_idx = specialists.index("Стоматолог")
            fake_probs[0, dental_idx] = 10.0
            return "Стоматолог", torch.softmax(fake_probs, dim=1)
        
    surgery_keywords = [
        "колено", "колени", "колен", "сустав", "суставы", "шея", "плечо", "плече",
        "локоть", "локте", "спина", "поясница", "бедро", "бедре", "голень",
        "стопа", "стопе", "кисть", "запястье", "палец", "пальце",
        "кость", "кости", "кост", "рука", "руке", "нога", "ноге",
        "ушиб", "перелом", "вывих", "растяжение", "травма", "отек", "отёк"
    ]

    for keyword in surgery_keywords:
        if keyword in text_lower:
            fake_probs = torch.zeros(1, len(specialists))
            surg_idx = specialists.index("Хирург")
            fake_probs[0, surg_idx] = 10.0
            return "Хирург", torch.softmax(fake_probs, dim=1)

    keyword_mapping = {
        "ЛОР": ["горло", "нос", "ухо", "отит", "синусит", "ринит", "миндалин", "гланд", "гортан"],
        "Пульмонолог": ["кашель", "одышк", "дыхани", "бронх", "легк", "астм", "хрип", "мокрот"],
        "Кардиолог": ["сердц", "груд", "давлен", "пульс", "аритм", "гипертон", "стенокард", "инфаркт"],
        "Гастроэнтеролог": ["живот", "тошн", "рвот", "желудок", "кишечн", "изжог", "гастрит", "панкреат"],
        "Невролог": ["голов", "нерв", "онемен", "судорог", "мигрен", "головокруж", "парез", "инсульт"],
        "Хирург": ["швы", "порез", "травм", "операц", "растян", "перелом", "ушиб", "рана", "рану", "шов", "швы", "колен", "коленк", "сустав", "шея", "плеч",
    "локт", "спин", "поясниц", "бедр", "голен", "стоп", "кист", "запяст",
    "палец", "пальц", "кость", "кости", "кост", "рук", "ног",
    "грыж", "отек", "отёк", "опух", "удар", "паден"],
        "Дерматолог": ["кож", "сып", "зуд", "покрасн", "шелуш", "акне", "экзем", "дермат", "прыщ"]
    }

    for doctor, keywords in keyword_mapping.items():
        for keyword in keywords:
            if keyword in text_lower:
                fake_probs = torch.zeros(1, len(specialists))
                doctor_idx = specialists.index(doctor)
                fake_probs[0, doctor_idx] = 10.0
                return doctor, torch.softmax(fake_probs, dim=1)

    try:
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen)
        padded = torch.tensor(padded, dtype=torch.long)

        with torch.no_grad():
            logits = model(padded)
            probs = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()

        return specialists[predicted_class], probs
    except Exception as e:
        print("Ошибка модели:", e)
        return "Терапевт", None


def render_page(result_html=""):
    return f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Маршрутизация пациента</title>

        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">

        <style>
        html {{
            overflow-x: hidden;
        }}

        body {{
            overflow-x: hidden;
        }}
        @media (max-width: 768px) {{
                .hero h1 {{
                    font-size: 26px;
                    line-height: 1.15;
                }}

                .hero p {{
                    font-size: 15px;
                    line-height: 1.45;
                }}

                label {{
                    font-size: 15px;
                }}

                .result, .warn {{
                    padding: 14px;
                    font-size: 15px;
                }}

                .hero {{
                    width: calc(100% - 16px);
                    margin: 8px 8px 10px 8px;
                    padding: 16px 14px;
                    border-radius: 16px;
                    box-sizing: border-box;
                }}

                .hero h1 {{
                    font-size: 24px;
                    line-height: 1.2;
                }}

                .hero p {{
                    font-size: 14px;
                    line-height: 1.45;
                }}

                .container {{
                    width: calc(100% - 16px);
                    margin: 0 8px 12px 8px;
                    padding: 16px 14px;
                    border-radius: 16px;
                    box-sizing: border-box;
                }}

                label {{
                    font-size: 14px;
                }}

                input, textarea {{
                    font-size: 16px;
                    padding: 14px 12px;
                    border-radius: 10px;
                }}

                textarea {{
                    min-height: 120px;
                }}

                button {{
                    width: 100%;
                    font-size: 16px;
                    padding: 13px 16px;
                    border-radius: 10px;
                }}

                .result, .warn {{
                    padding: 14px;
                    font-size: 14px;
                }}

                ul {{
                    padding-left: 18px;
                }}
            }}
            body {{
                font-family: 'IBM Plex Sans', Arial, sans-serif;
                margin: 0;
                padding: 0;
                min-height: 100vh;
                overflow-x: hidden;
                background:
                    linear-gradient(rgba(16, 36, 58, 0.55), rgba(16, 36, 58, 0.55)),
                    url("/static/hospital_bg.jpg") center center / cover no-repeat fixed;
            }}

            .hero {{
                max-width: 900px;
                margin: 36px auto 18px auto;
                padding: 26px 30px;
                border-radius: 24px;
                background: rgba(255, 255, 255, 0.14);
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.14);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                color: white;
            }}

            .hero h1 {{
                margin: 0 0 10px 0;
                font-size: 34px;
                font-weight: 700;
                letter-spacing: -0.03em;
                color: white;
            }}

            .hero p {{
                margin: 0;
                font-size: 16px;
                line-height: 1.5;
                color: rgba(255, 255, 255, 0.88);
            }}

            .container {{
                max-width: 820px;
                margin: 0 auto 40px auto;
                background: rgba(255, 255, 255, 0.92);
                padding: 34px;
                border-radius: 22px;
                box-shadow: 0 18px 50px rgba(0, 0, 0, 0.18);
                backdrop-filter: blur(8px);
                -webkit-backdrop-filter: blur(8px);
            }}

            label {{
                display: block;
                margin-top: 15px;
                margin-bottom: 6px;
                font-weight: 600;
                color: #183b56;
            }}

            input, textarea {{
                width: 100%;
                padding: 12px 14px;
                border: 1px solid #d9e2ec;
                border-radius: 12px;
                font-size: 15px;
                box-sizing: border-box;
                background: rgba(255, 255, 255, 0.92);
                font-family: 'IBM Plex Sans', Arial, sans-serif;
            }}

            textarea {{
                min-height: 140px;
                resize: vertical;
            }}

            button {{
                width: 100%;
                margin-top: 20px;
                background: linear-gradient(135deg, #2d8cff, #1f6fd1);
                color: white;
                border: none;
                padding: 13px 22px;
                border-radius: 12px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                box-shadow: 0 8px 20px rgba(31, 122, 224, 0.28);
                transition: 0.2s ease;
                font-family: 'IBM Plex Sans', Arial, sans-serif;
            }}

            button:hover {{
                transform: translateY(-1px);
                background: linear-gradient(135deg, #2a7fe8, #185fb5);
            }}

            .result {{
                margin-top: 25px;
                padding: 18px;
                background: #eef6ff;
                border-left: 5px solid #1f7ae0;
                border-radius: 10px;
            }}

            .warn {{
                margin-top: 25px;
                padding: 18px;
                background: #fff4e5;
                border-left: 5px solid #ff9800;
                border-radius: 10px;
            }}

            ul {{
                padding-left: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="hero">
            <h1>Система интеллектуальной маршрутизации пациента</h1>
            <p>Цифровой сервис предварительной маршрутизации пациента на основе анализа жалоб и симптомов.</p>
        </div>

        <div class="container">
            <form method="get" action="/predict">
                <label>ФИО пациента</label>
                <input type="text" name="fio" placeholder="Впишите сюда своё ФИО" required>

                <label>Telegram / ID пациента</label>
                <input type="text" name="telegram" value="web_user" placeholder="Впишите свой ник в Telegram или ID">

                <label>Жалоба / симптомы</label>
                <textarea name="complaint" placeholder="Подробно опишите свою жалобу" required></textarea>

                <button type="submit">Определить специалиста</button>
            </form>

            {result_html}

            <div class="warn">
                <b>Важно:</b> система используется для предварительной маршрутизации и не заменяет медицинскую консультацию.
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
def home():
    return render_page()


@app.get("/predict", response_class=HTMLResponse)
def predict(fio: str = "", telegram: str = "web_user", complaint: str = ""):
    if not fio.strip() or not complaint.strip():
        result_html = """
        <div class="warn">
            <b>Ошибка:</b> заполните ФИО и жалобу.
        </div>
        """
        return render_page(result_html)

    doctor, probs = predict_complaint(complaint)

    if logger is not None:
        row_idx = logger.create_case(fio, telegram)
        logger.update_case(row_idx, complaint=complaint, doctor=doctor)

    probs_html = ""
    if probs is not None:
        pairs = list(zip(specialists, probs[0].tolist()))
        pairs.sort(key=lambda x: x[1], reverse=True)

        probs_html = "<h3>Вероятности по специалистам:</h3><ul>"
        for spec, prob in pairs:
            probs_html += f"<li><b>{spec}</b>: {prob:.2%}</li>"
        probs_html += "</ul>"

    result_html = f"""
    <div class="result">
        <h2>Результат</h2>
        <p><b>ФИО:</b> {fio}</p>
        <p><b>Жалоба:</b> {complaint}</p>
        <p><b>Рекомендуемый специалист:</b> {doctor}</p>
        {probs_html}
    </div>
    """

    return render_page(result_html)