import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

try:
    from secure_excel_logger import SecureExcelLogger
except ImportError:
    SecureExcelLogger = None

app = FastAPI(title="Маршрутизация пациента")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

print("Загрузка трансформера RuBioRoBERTa")

tokenizer = AutoTokenizer.from_pretrained("tokenizer")
model = AutoModelForSequenceClassification.from_pretrained("doctor_model")
model.eval()

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True
)

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

if APP_MODE == "local" and SecureExcelLogger is not None:
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
    print("ℹ️ Demo mode или модуль логгера недоступен")

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
            return "Стоматолог", None

    surgery_keywords = [
        "колено", "колени", "колен", "коленк", "сустав", "суставы", "шея", "плечо", "плече",
        "локоть", "локте", "спина", "поясница", "бедро", "бедре", "голень",
        "стопа", "стопе", "кисть", "запястье", "палец", "пальце",
        "кость", "кости", "кост", "рука", "руке", "нога", "ноге",
        "ушиб", "перелом", "вывих", "растяжение", "травма", "отек", "отёк"
    ]
    for keyword in surgery_keywords:
        if keyword in text_lower:
            return "Хирург", None

    keyword_mapping = {
        "ЛОР": ["горло", "нос", "ухо", "отит", "синусит", "ринит", "миндалин", "гланд", "гортан"],
        "Пульмонолог": ["кашель", "одышк", "дыхани", "бронх", "легк", "астм", "хрип", "мокрот"],
        "Кардиолог": ["сердц", "груд", "давлен", "пульс", "аритм", "гипертон", "стенокард", "инфаркт"],
        "Гастроэнтеролог": ["живот", "тошн", "рвот", "желудок", "кишечн", "изжог", "гастрит", "панкреат"],
        "Невролог": ["голов", "нерв", "онемен", "судорог", "мигрен", "головокруж", "парез", "инсульт"],
        "Хирург": ["швы", "порез", "травм", "операц", "растян", "перелом", "ушиб", "рана", "рану", "шов", "швы", "колен", "коленк", "сустав", "шея", "плеч", "локт", "спин", "поясниц", "бедр", "голен", "стоп", "кист", "запяст", "палец", "пальц", "кость", "кости", "кост", "рук", "ног", "грыж", "отек", "отёк", "опух", "удар", "паден"],
        "Дерматолог": ["кож", "сып", "зуд", "покрасн", "шелуш", "акне", "экзем", "дермат", "прыщ"]
    }

    for doctor, keywords in keyword_mapping.items():
        for keyword in keywords:
            if keyword in text_lower:
                return doctor, None

    # === НОВАЯ часть — трансформер 💕 ===
    try:
        result = classifier(text)[0]
        scores = [item['score'] for item in result]
        predicted_idx = torch.tensor(scores).argmax().item()
        doctor = specialists[predicted_idx]
        return doctor, torch.tensor(scores)
    except Exception as e:
        print("Ошибка модели:", e)
        return "Терапевт", None

def get_schedule_for_doctor(doctor):
    schedule = {
        "Терапевт": ["09:00", "10:00", "11:30", "13:00", "15:00", "17:00"],
        "ЛОР": ["10:00", "11:30", "13:30", "15:30"],
        "Пульмонолог": ["09:30", "11:00", "14:00", "16:00"],
        "Кардиолог": ["09:00", "10:30", "12:30", "15:00"],
        "Гастроэнтеролог": ["10:00", "12:00", "14:30", "16:30"],
        "Невролог": ["09:30", "11:30", "13:30", "16:00"],
        "Хирург": ["09:00", "11:00", "13:00", "15:00"],
        "Дерматолог": ["10:00", "12:30", "14:30", "17:00"],
        "Окулист": ["08:30", "10:00", "12:00", "15:30"],
        "Стоматолог": ["09:00", "10:30", "12:30", "14:30"]
    }
    return schedule.get(doctor, ["10:00", "12:00", "14:00"])

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

            input, textarea, select {{
                width: 100%;
                padding: 12px 14px;
                border: 1px solid #d9e2ec;
                border-radius: 12px;
                font-size: 15px;
                box-sizing: border-box;
                background: rgba(255, 255, 255, 0.92);
                font-family: 'IBM Plex Sans', Arial, sans-serif;
            }}

            input::placeholder,
            textarea::placeholder {{
                color: #9aa7b5;
                opacity: 1;
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

            .booking-block {{
                margin-top: 22px;
                padding: 18px;
                background: rgba(248, 251, 255, 0.96);
                border: 1px solid #d9e7f5;
                border-radius: 14px;
            }}

            .booking-title {{
                margin: 0 0 10px 0;
                color: #183b56;
                font-size: 22px;
            }}

            .booking-text {{
                margin-bottom: 14px;
                color: #486581;
                line-height: 1.45;
            }}

            .doctor-buttons {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-bottom: 16px;
            }}

            .doctor-choice {{
                flex: 1 1 220px;
                margin-top: 0;
            }}

            .slot-grid {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 12px;
            }}

            .slot-btn {{
                background: #eef6ff;
                color: #1f4e79;
                border: 1px solid #cfe0f2;
                padding: 10px 14px;
                border-radius: 10px;
                font-size: 15px;
                font-weight: 600;
            }}

            .slot-btn:hover {{
                background: #dcecff;
            }}

            .booking-actions {{
                margin-top: 16px;
            }}

            ul {{
                padding-left: 20px;
            }}

            @media (max-width: 768px) {{
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

                input, textarea, select {{
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

                .doctor-buttons {{
                    flex-direction: column;
                }}

                .doctor-choice {{
                    width: 100%;
                }}

                .slot-btn {{
                    width: 100%;
                    text-align: center;
                }}

                ul {{
                    padding-left: 18px;
                }}
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
                <input type="text" name="telegram" placeholder="Впишите свой ник в Telegram или ID">

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

    today = date.today().isoformat()
    recommended_slots = get_schedule_for_doctor(doctor)
    therapist_slots = get_schedule_for_doctor("Терапевт")

    recommended_slots_html = "".join(
        [f'<button type="button" class="slot-btn">{slot}</button>' for slot in recommended_slots]
    )
    therapist_slots_html = "".join(
        [f'<button type="button" class="slot-btn">{slot}</button>' for slot in therapist_slots]
    )

    result_html = f"""
    <div class="result">
        <h2>Результат</h2>
        <p><b>ФИО:</b> {fio}</p>
        <p><b>Жалоба:</b> {complaint}</p>
        <p><b>Рекомендуемый специалист:</b> {doctor}</p>
        {probs_html}

        <div class="booking-actions">
            <button type="button" onclick="showBookingStep1()">
                Записаться к врачу
            </button>
        </div>
    </div>

    <div id="booking-step-1" class="booking-block" style="display:none;">
        <h2 class="booking-title">Выбор специалиста</h2>
        <p class="booking-text">
            Выберите врача для предварительной записи.
        </p>

        <div class="doctor-buttons">
            <button type="button" class="doctor-choice" onclick="showDoctorBooking('therapist-booking')">
                Терапевт
            </button>
            <button type="button" class="doctor-choice" onclick="showDoctorBooking('recommended-booking')">
                {doctor}
            </button>
        </div>
    </div>

    <div id="therapist-booking" class="booking-block" style="display:none;">
        <h2 class="booking-title">Запись к терапевту</h2>
        <form method="get" action="/book">
            <input type="hidden" name="fio" value="{fio}">
            <input type="hidden" name="telegram" value="{telegram}">
            <input type="hidden" name="complaint" value="{complaint}">
            <input type="hidden" name="doctor" value="Терапевт">

            <label>Дата записи к терапевту</label>
            <input type="date" name="visit_date" min="{today}" required>

            <label>Время записи</label>
            <select name="visit_time" required>
                <option value="">Выберите время</option>
                {"".join([f'<option value="{slot}">{slot}</option>' for slot in therapist_slots])}
            </select>

            <button type="submit">Подтвердить запись к терапевту</button>
        </form>
    </div>

    <div id="recommended-booking" class="booking-block" style="display:none;">
        <h2 class="booking-title">Запись к врачу: {doctor}</h2>
        <form method="get" action="/book">
            <input type="hidden" name="fio" value="{fio}">
            <input type="hidden" name="telegram" value="{telegram}">
            <input type="hidden" name="complaint" value="{complaint}">
            <input type="hidden" name="doctor" value="{doctor}">

            <label>Дата записи к врачу: {doctor}</label>
            <input type="date" name="visit_date" min="{today}" required>

            <label>Время записи</label>
            <select name="visit_time" required>
                <option value="">Выберите время</option>
                {"".join([f'<option value="{slot}">{slot}</option>' for slot in recommended_slots])}
            </select>

            <button type="submit">Подтвердить запись к врачу</button>
        </form>
    </div>

    <script>
        function showBookingStep1() {{
            document.getElementById('booking-step-1').style.display = 'block';
            document.getElementById('booking-step-1').scrollIntoView({{behavior: 'smooth', block: 'start'}});
        }}

        function showDoctorBooking(blockId) {{
            document.getElementById('therapist-booking').style.display = 'none';
            document.getElementById('recommended-booking').style.display = 'none';

            document.getElementById(blockId).style.display = 'block';
            document.getElementById(blockId).scrollIntoView({{behavior: 'smooth', block: 'start'}});
        }}
    </script>
    """

    return render_page(result_html)