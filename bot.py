import torch
import numpy as np
import pickle
import telebot
import telebot.apihelper as apihelper
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
from secure_excel_logger import SecureExcelLogger


print("🤖 Загрузка модели для Telegram-бота...")

try:

    with open('model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print("✅ Метаданные загружены")
    
except FileNotFoundError:
    print("❌ Файлы модели не найдены!")
    print("Сначала запустите: python main.py")
    exit()

maxlen = metadata['maxlen']
specialists = metadata['specialists']
vocab_size = metadata['vocab_size']
word_index = metadata['word_index']

print(f"📊 Специалисты: {', '.join(specialists)}")
print(f"📖 Размер словаря: {vocab_size} слов")

class SimpleTokenizer:
    def __init__(self, word_index):
        self.word_index = word_index
    
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = [self.word_index.get(word, 0) for word in text.lower().split()]
            sequences.append(seq)
        return sequences

tokenizer = SimpleTokenizer(word_index)

def pad_sequences(sequences, maxlen):
    padded = []
    for seq in sequences:
        if len(seq) > maxlen:
            seq = seq[:maxlen]
        else:
            seq = [0]*(maxlen - len(seq)) + seq
        padded.append(seq)
    return np.array(padded)

class DoctorNet(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_classes=8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size + 1, embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        pooled = lstm_out.mean(dim=1)
        out = self.fc(pooled)
        return out

model = DoctorNet(vocab_size)
model.load_state_dict(torch.load('doctor_model.pth'))
model.eval()
print("✅ Модель загружена и готова к работе!")

def predict_complaint(text):
    """Предсказание врача по жалобе с приоритетом ключевых слов"""
    
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
    
    keyword_mapping = {
        "ЛОР": ["горло", "нос", "ухо", "отит", "синусит", "ринит", "миндалин", "гланд", "гортан"],
        "Пульмонолог": ["кашель", "одышк", "дыхани", "бронх", "легк", "астм", "хрип", "мокрот"],
        "Кардиолог": ["сердц", "груд", "давлен", "пульс", "аритм", "гипертон", "стенокард", "инфаркт"],
        "Гастроэнтеролог": ["живот", "тошн", "рвот", "желудок", "кишечн", "изжог", "гастрит", "панкреат"],
        "Невролог": ["голов", "нерв", "онемен", "судорог", "мигрен", "головокруж", "парез", "инсульт"],
        "Хирург": ["порез", "травм", "операц", "растян", "перелом", "ушиб", "рану", "шов", "швы"],
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
        print(f"Ошибка модели: {e}")
        return "Терапевт", None

TOKEN = "7961825225:AAGJxJJiQqaN9z7dML7dyxq_5gQ0yl65ClA"

apihelper.CONNECT_TIMEOUT = 30
apihelper.READ_TIMEOUT = 30

apihelper.proxy = {
    "http": "socks5://142.54.239.1",
    "https": "socks5://142.54.239.1"
}

bot = telebot.TeleBot(TOKEN)

logger = SecureExcelLogger(
    excel_path="cases.xlsx",
    key_path="secret.key"
)

user_sessions = {}

def get_telegram_tag(message) -> str:
    u = message.from_user
    if getattr(u, "username", None):
        return f"@{u.username}"
    return f"tg_id:{u.id}"

def create_keyboard():
    markup = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)

    buttons = [
        "📝 Новая жалоба",
        "ℹ️ Помощь"
    ]

    for btn in buttons:
        markup.add(KeyboardButton(btn))

    return markup

@bot.message_handler(commands=['start'])
def start_command(message):
    user_sessions[message.from_user.id] = {"stage": "await_fio"}
    bot.send_message(
        message.chat.id,
        "👋 Привет! Введите ваше ФИО одним сообщением:",
        reply_markup=create_keyboard()
    )


@bot.message_handler(commands=['new'])
def new_case(message):
    sess = user_sessions.get(message.from_user.id)
    if not sess or "fio" not in sess:
        user_sessions[message.from_user.id] = {"stage": "await_fio"}
        bot.send_message(message.chat.id, "Введите ваше ФИО одним сообщением:")
        return

    sess["stage"] = "await_complaint"
    telegram = get_telegram_tag(message)
    sess["telegram"] = telegram
    sess["row_idx"] = logger.create_case(sess["fio"], telegram)

    bot.send_message(message.chat.id, "✅ Ок. Теперь введите вашу жалобу (симптомы) одним сообщением:")

@bot.message_handler(commands=['help', 'info'])
def help_command(message):
    """Обработчик команды /help"""
    help_text = (
        "🩺 *Доступные специалисты:*\n\n"
        "• *ЛОР* - уши, горло, нос, пазухи\n"
        "• *Пульмонолог* - легкие, дыхание, кашель\n"  
        "• *Кардиолог* - сердце, давление, сосуды\n"
        "• *Невролог* - голова, спина, нервы, онемение, головокружение\n"
        "• *Хирург* - раны, травмы, операции, боль в животе, опухоли\n"
        "• *Стоматолог* - зубы, десны, боль в зубах, чувствительность, кариеси\n"
        "• *Дерматолог* - кожа, сыпь, зуд, покраснение, шелушение\n"
        "• *Гастроэнтеролог* - желудок, кишечник\n\n"
        "💡 *Совет:* опишите симптомы подробнее."
    )
    
    bot.send_message(message.chat.id, help_text, parse_mode='Markdown')

@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    text = (message.text or "").strip()

    if text == "📝 Новая жалоба":
        new_case(message)
        return

    if text == "ℹ️ Помощь":
        help_command(message)
        return

    user_id = message.from_user.id
    text = (message.text or "").strip()
    if not text:
        bot.send_message(message.chat.id, "Напишите текстом 🙂")
        return

    sess = user_sessions.get(user_id)

    if not sess:
        user_sessions[user_id] = {"stage": "await_fio"}
        bot.send_message(message.chat.id, "Введите ваше ФИО одним сообщением:")
        return

    stage = sess.get("stage")

    if stage == "await_fio":
        fio = text
        telegram = get_telegram_tag(message)

        row_idx = logger.create_case(fio, telegram)
        user_sessions[user_id] = {
            "stage": "await_complaint",
            "fio": fio,
            "telegram": telegram,
            "row_idx": row_idx
        }

        bot.send_message(message.chat.id, "✅ ФИО сохранено. Теперь введите вашу жалобу (симптомы):")
        return

    if stage == "await_complaint":
        complaint = text

        bot.send_chat_action(message.chat.id, 'typing')

        doctor, probs = predict_complaint(complaint)

        logger.update_case(sess["row_idx"], complaint=complaint, doctor=doctor)

        result_text = (
            f"📝 *Жалоба:* {complaint}\n\n"
            f"👨‍⚕️ *Рекомендуемый врач:* *{doctor}*\n"
        )

        bot.send_message(message.chat.id, result_text, parse_mode='Markdown', reply_markup=create_keyboard())

        user_sessions[user_id]["stage"] = "done"
        bot.send_message(message.chat.id, "Если хотите оформить ещё одно обращение — напишите /new")
        return

    if stage == "done":
        bot.send_message(message.chat.id, "Для нового обращения напишите /new (или /start чтобы заново ввести ФИО).")
        return

import time

while True:
    try:
        print("Бот запускается...")
        bot.polling(non_stop=True, interval=2, timeout=30)
    except Exception as e:
        print("Ошибка:", e)
        time.sleep(5)