import os
import torch
import pickle
import numpy as np
from datetime import date
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

try:
    from secure_excel_logger import SecureExcelLogger
except ImportError:
    SecureExcelLogger = None

app = FastAPI(title="Маршрутизация пациента")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

APP_MODE = os.getenv("APP_MODE", "demo")
classifier = None

if APP_MODE == "production":
    print("🚀 Production mode: загрузка локальной модели...")

    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained("doctor_model")
    model.eval()

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True
    )
else:
    print("ℹ️ Demo mode: модель не загружается")

try:
    with open("model_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    specialists = metadata.get("specialists", [
        "ЛОР", "Пульмонолог", "Кардиолог", "Гастроэнтеролог",
        "Невролог", "Хирург", "Стоматолог", "Дерматолог", "Окулист"
    ])
except:
    specialists = ["ЛОР", "Пульмонолог", "Кардиолог", "Гастроэнтеролог",
                   "Невролог", "Хирург", "Стоматолог", "Дерматолог", "Окулист"]

logger = None
if os.getenv("APP_MODE", "local") == "local" and SecureExcelLogger is not None:
    try:
        logger = SecureExcelLogger("cases.xlsx", "secret.key")
        print("✅ Логгер подключён")
    except:
        pass

def predict_complaint(text):
    # Быстрые ключевые слова
    text_lower = text.lower()
    if any(w in text_lower for w in ["зуб", "зубы", "десн", "стоматолог"]):
        return "Стоматолог", None
    if any(w in text_lower for w in ["колено", "перелом", "ушиб", "травм", "сустав"]):
        return "Хирург", None
    if any(w in text_lower for w in ["каш", "одыш", "бронх", "легк", "мокрот"]):
        return "Пульмонолог", None
    if any(w in text_lower for w in ["сердц", "груд", "давлен", "аритм", "пульс"]):
        return "Кардиолог", None
    if any(w in text_lower for w in ["ухо", "горло", "нос", "ангин", "насморк"]):
        return "ЛОР", None
    if any(w in text_lower for w in ["голов", "мигр", "онем", "нерв", "головокруж"]):
        return "Невролог", None
    if any(w in text_lower for w in ["живот", "тошн", "рвот", "изжог", "желуд"]):
        return "Гастроэнтеролог", None
    if any(w in text_lower for w in ["сып", "кож", "зуд", "пятн"]):
        return "Дерматолог", None
    if any(w in text_lower for w in ["глаз", "зрен", "веки"]):
        return "Окулист", None

    try:
        result = classifier(text)[0]
        scores = [item['score'] for item in result]
        pred_idx = torch.tensor(scores).argmax().item()
        doctor = specialists[pred_idx]
        return doctor, torch.tensor(scores)
    except:
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
        <title>MEDREG AI — Intelligent Patient Routing</title>

        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">

        <style>
            * {{
                box-sizing: border-box;
            }}

            html {{
                overflow-x: hidden;
            }}

            body {{
                margin: 0;
                min-height: 100vh;
                overflow-x: hidden;
                font-family: 'IBM Plex Sans', Arial, sans-serif;
                color: #ffffff;
                background:
                    radial-gradient(circle at 20% 15%, rgba(0, 229, 255, 0.22), transparent 32%),
                    radial-gradient(circle at 85% 20%, rgba(45, 140, 255, 0.25), transparent 30%),
                    linear-gradient(rgba(4, 14, 28, 0.80), rgba(4, 14, 28, 0.90)),
                    url("/static/hospital_bg.jpg") center center / cover no-repeat fixed;
            }}

            body::before {{
                content: "";
                position: fixed;
                inset: 0;
                pointer-events: none;
                background-image:
                    linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
                background-size: 44px 44px;
                mask-image: linear-gradient(to bottom, rgba(0,0,0,0.8), transparent);
            }}

            .page {{
                width: min(1180px, calc(100% - 32px));
                margin: 0 auto;
                padding: 34px 0 48px;
            }}

            .topbar {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 34px;
            }}

            .brand {{
                display: flex;
                align-items: center;
                gap: 12px;
            }}

            .logo {{
                width: 46px;
                height: 46px;
                border-radius: 16px;
                background: linear-gradient(135deg, #00e5ff, #2d8cff);
                display: grid;
                place-items: center;
                color: #03101f;
                font-weight: 800;
                box-shadow: 0 0 28px rgba(0, 229, 255, 0.45);
            }}

            .brand-title {{
                font-size: 22px;
                font-weight: 800;
                letter-spacing: -0.04em;
            }}

            .brand-subtitle {{
                font-size: 13px;
                color: rgba(255,255,255,0.62);
                margin-top: 2px;
            }}

            .status-pill {{
                padding: 10px 14px;
                border: 1px solid rgba(0, 229, 255, 0.28);
                border-radius: 999px;
                background: rgba(4, 20, 38, 0.58);
                backdrop-filter: blur(14px);
                color: #bff7ff;
                font-size: 13px;
                box-shadow: inset 0 0 20px rgba(0, 229, 255, 0.06);
            }}

            .hero-grid {{
                display: grid;
                grid-template-columns: 1.05fr 0.95fr;
                gap: 24px;
                align-items: stretch;
            }}

            .hero-card, .form-card, .side-card {{
                border: 1px solid rgba(140, 220, 255, 0.20);
                background: linear-gradient(180deg, rgba(8, 28, 52, 0.78), rgba(5, 18, 35, 0.86));
                box-shadow: 0 28px 80px rgba(0, 0, 0, 0.36);
                backdrop-filter: blur(18px);
                border-radius: 28px;
            }}

            .hero-card {{
                padding: 34px;
                position: relative;
                overflow: hidden;
            }}

            .hero-card::after {{
                content: "";
                position: absolute;
                width: 280px;
                height: 280px;
                right: -90px;
                top: -90px;
                border-radius: 50%;
                background: radial-gradient(circle, rgba(0,229,255,0.26), transparent 68%);
            }}

            .eyebrow {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 8px 12px;
                border-radius: 999px;
                background: rgba(0, 229, 255, 0.10);
                border: 1px solid rgba(0, 229, 255, 0.22);
                color: #aef7ff;
                font-size: 13px;
                margin-bottom: 18px;
            }}

            h1 {{
                margin: 0;
                font-size: clamp(34px, 5vw, 62px);
                line-height: 0.95;
                letter-spacing: -0.06em;
            }}

            .gradient-text {{
                background: linear-gradient(135deg, #ffffff, #9eeeff 45%, #2d8cff);
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
            }}

            .hero-text {{
                margin: 18px 0 0;
                max-width: 620px;
                color: rgba(255,255,255,0.76);
                font-size: 17px;
                line-height: 1.65;
            }}

            .metrics {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 12px;
                margin-top: 26px;
            }}

            .metric {{
                padding: 16px;
                border-radius: 18px;
                background: rgba(255,255,255,0.055);
                border: 1px solid rgba(255,255,255,0.10);
            }}

            .metric b {{
                display: block;
                font-size: 24px;
                color: #ffffff;
            }}

            .metric span {{
                display: block;
                margin-top: 4px;
                font-size: 12px;
                color: rgba(255,255,255,0.58);
            }}

            .form-card {{
                padding: 26px;
            }}

            .form-title {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 18px;
            }}

            .form-title h2 {{
                margin: 0;
                font-size: 24px;
                letter-spacing: -0.03em;
            }}

            .ai-badge {{
                color: #06182a;
                background: linear-gradient(135deg, #00e5ff, #ffffff);
                border-radius: 999px;
                padding: 7px 10px;
                font-size: 12px;
                font-weight: 800;
            }}

            label {{
                display: block;
                margin: 15px 0 7px;
                color: rgba(255,255,255,0.82);
                font-weight: 600;
                font-size: 14px;
            }}

            input, textarea, select {{
                width: 100%;
                border: 1px solid rgba(145, 210, 255, 0.24);
                background: rgba(2, 12, 24, 0.62);
                color: #ffffff;
                border-radius: 16px;
                padding: 14px 15px;
                font-size: 16px;
                font-family: 'IBM Plex Sans', Arial, sans-serif;
                outline: none;
                transition: 0.18s ease;
            }}

            input:focus, textarea:focus, select:focus {{
                border-color: rgba(0, 229, 255, 0.72);
                box-shadow: 0 0 0 4px rgba(0, 229, 255, 0.10);
            }}

            input::placeholder, textarea::placeholder {{
                color: rgba(255,255,255,0.38);
            }}

            textarea {{
                min-height: 140px;
                resize: vertical;
            }}

            button {{
                width: 100%;
                margin-top: 20px;
                border: none;
                border-radius: 16px;
                padding: 15px 20px;
                background: linear-gradient(135deg, #00e5ff, #2d8cff);
                color: #03101f;
                font-size: 16px;
                font-weight: 800;
                cursor: pointer;
                font-family: 'IBM Plex Sans', Arial, sans-serif;
                box-shadow: 0 16px 40px rgba(0, 170, 255, 0.26);
                transition: 0.2s ease;
            }}

            button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 20px 48px rgba(0, 170, 255, 0.35);
            }}

            .side-stack {{
                display: grid;
                gap: 16px;
                margin-top: 24px;
                grid-template-columns: repeat(4, 1fr);
            }}

            .side-card {{
                padding: 18px;
                min-height: 118px;
            }}

            .side-card small {{
                color: #8eeeff;
                font-weight: 700;
            }}

            .side-card p {{
                margin: 8px 0 0;
                color: rgba(255,255,255,0.68);
                font-size: 14px;
                line-height: 1.45;
            }}

            .result {{
                margin-top: 22px;
                padding: 20px;
                border-radius: 20px;
                background: linear-gradient(180deg, rgba(0, 229, 255, 0.12), rgba(45, 140, 255, 0.10));
                border: 1px solid rgba(0, 229, 255, 0.26);
            }}

            .result h2 {{
                margin-top: 0;
                letter-spacing: -0.03em;
            }}

            .result p, .result li {{
                color: rgba(255,255,255,0.82);
            }}

            .warn {{
                margin-top: 20px;
                padding: 16px;
                border-radius: 18px;
                background: rgba(255, 194, 102, 0.10);
                border: 1px solid rgba(255, 194, 102, 0.28);
                color: rgba(255,255,255,0.78);
                line-height: 1.5;
            }}

            .booking-block {{
                margin-top: 18px;
                padding: 20px;
                border-radius: 20px;
                background: rgba(2, 12, 24, 0.56);
                border: 1px solid rgba(145, 210, 255, 0.18);
            }}

            .booking-title {{
                margin: 0 0 10px;
                color: #ffffff;
                letter-spacing: -0.03em;
            }}

            .booking-text {{
                color: rgba(255,255,255,0.66);
            }}

            .doctor-buttons {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
            }}

            select option {{
                color: #06182a;
            }}

            .feature-strip {{
                margin-top: 22px;
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                justify-content: center;
                color: rgba(255,255,255,0.70);
                font-size: 13px;
            }}

            .feature-strip span {{
                padding: 9px 12px;
                border-radius: 999px;
                border: 1px solid rgba(255,255,255,0.10);
                background: rgba(255,255,255,0.055);
            }}

            @media (max-width: 900px) {{
                .hero-grid {{
                    grid-template-columns: 1fr;
                }}

                .side-stack {{
                    grid-template-columns: 1fr 1fr;
                }}
            }}

            @media (max-width: 560px) {{
                .page {{
                    width: min(100% - 18px, 1180px);
                    padding-top: 14px;
                }}

                .topbar {{
                    align-items: flex-start;
                    gap: 12px;
                    flex-direction: column;
                }}

                .hero-card, .form-card {{
                    border-radius: 22px;
                    padding: 22px;
                }}

                .metrics {{
                    grid-template-columns: 1fr;
                }}

                .side-stack {{
                    grid-template-columns: 1fr;
                }}

                .doctor-buttons {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>

    <body>
        <main class="page">
            <header class="topbar">
                <div class="brand">
                    <div class="logo">M</div>
                    <div>
                        <div class="brand-title">MEDREG AI</div>
                        <div class="brand-subtitle">Intelligent Neural Patient Routing Platform</div>
                    </div>
                </div>
                <div class="status-pill">● Cloud demo online</div>
            </header>

            <section class="hero-grid">
                <div class="hero-card">
                    <div class="eyebrow">⚕ Neural medical routing system</div>
                    <h1>
                        <span class="gradient-text">Система интеллектуальной маршрутизации пациента</span>
                    </h1>
                    <p class="hero-text">
                        MEDREG AI анализирует жалобы пациента, классифицирует симптомы и предлагает профильного специалиста
                        для предварительной медицинской маршрутизации.
                    </p>

                    <div class="metrics">
                        <div class="metric">
                            <b>9</b>
                            <span>медицинских направлений</span>
                        </div>
                        <div class="metric">
                            <b>NLP</b>
                            <span>анализ жалоб</span>
                        </div>
                        <div class="metric">
                            <b>AI</b>
                            <span>нейросетевая рекомендация</span>
                        </div>
                    </div>
                </div>

                <div class="form-card">
                    <div class="form-title">
                        <h2>Patient Routing Console</h2>
                        <span class="ai-badge">AI TRIAGE</span>
                    </div>

                    <form method="get" action="/predict">
                        <label>ФИО пациента</label>
                        <input type="text" name="fio" placeholder="Например: Иванов Иван Иванович" required>

                        <label>Telegram / ID пациента</label>
                        <input type="text" name="telegram" placeholder="@username или ID пациента">

                        <label>Жалоба / симптомы</label>
                        <textarea name="complaint" placeholder="Опишите симптомы: боль, кашель, температура, одышка, сыпь..." required></textarea>

                        <button type="submit">Определить специалиста →</button>
                    </form>

                    {result_html}

                    <div class="warn">
                        <b>Важно:</b> система предназначена для предварительной маршрутизации пациента и не заменяет консультацию врача.
                    </div>
                </div>
            </section>

            <section class="side-stack">
                <div class="side-card">
                    <small>01 / Neural Complaint Analysis</small>
                    <p>Обработка свободного текста жалобы пациента.</p>
                </div>
                <div class="side-card">
                    <small>02 / Symptom Classification</small>
                    <p>Классификация симптомов по медицинским направлениям.</p>
                </div>
                <div class="side-card">
                    <small>03 / Recommended Specialist</small>
                    <p>Вывод рекомендуемого специалиста и вероятностей.</p>
                </div>
                <div class="side-card">
                    <small>04 / Secure Patient Logging</small>
                    <p>Подготовка к защищённому журналированию обращений.</p>
                </div>
            </section>

            <div class="feature-strip">
                <span>AI Triage</span>
                <span>NLP Complaint Parsing</span>
                <span>Medical Routing</span>
                <span>Privacy Protection</span>
                <span>Cloud Deployable</span>
            </div>
        </main>
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
