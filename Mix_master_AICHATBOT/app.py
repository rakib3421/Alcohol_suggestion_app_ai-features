import os
import io
import uuid
from flask import Flask, request, jsonify, session
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from PIL import Image
import mysql.connector
import google.generativeai as genai
from flask_cors import CORS  # Import CORS to handle cross-origin requests

# Load API keys
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your_fallback_secret_key")
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Enable CORS for all routes (for development purposes)
CORS(app)


def get_db():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB"),
    )


# Get current session_id
def get_session_id():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


# Load chat history for current session
def load_chat_history():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute(
        "SELECT role, message FROM chat_history WHERE session_id = %s AND source = 'text' ORDER BY id ASC",
        (get_session_id(),),
    )
    history = cursor.fetchall()
    db.close()
    return history


# Save message to DB with session_id
def save_message(role, message, source="text"):
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO chat_history (role, message, source, session_id) VALUES (%s, %s, %s, %s)",
        (role, message, source, get_session_id()),
    )
    db.commit()
    db.close()


# Remove duplicate lines in Gemini response
def remove_duplicates(text):
    lines = []
    seen_labels = set()

    for line in text.split("\n"):
        if not line.strip():
            continue
        label = line.split(":")[0].strip()
        if label in seen_labels:
            continue
        seen_labels.add(label)
        lines.append(line.strip())

    return "\n".join(lines)


@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    save_message("user", user_input)

    history = load_chat_history()
    chat = genai.GenerativeModel("gemini-2.0-flash").start_chat(
        history=[
            {"role": msg["role"], "parts": [{"text": msg["message"]}]}
            for msg in history
        ]
    )
    response = chat.send_message(user_input)

    cleaned = remove_duplicates(response.text)
    save_message("model", cleaned)
    return jsonify({"response": cleaned, "history": history})


@app.route("/upload", methods=["POST"])
def upload_image():
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    image = Image.open(path).convert("RGB")
    byte_stream = io.BytesIO()
    image.save(byte_stream, format="JPEG")
    image_bytes = byte_stream.getvalue()

    prompt = (
        "You're an expert bottle identification assistant. "
        "Please strictly return ONLY these 12 details in this exact format, using emojis, without any extra sentences. "
        "If a field is unknown, write 'Not specified'.\n\n"
        "🔍 Identified alcohol:\n"
        "🌍 Origin:\n"
        "🍸 Alcohol Content:\n"
        "🌾 Main Ingredient:\n"
        "👅 Tasting Notes:\n"
        "🔹 Similar kind of alcohol (at least 3):\n"
        "🔗 Want to mix a cocktail? Try a recipe:\n"
        "✨ AI Bot Interactive Features:\n"
        "📊 Confidence Level:\n"
        "🏷 Brand Logo & History:\n"
        "🎥 YouTube Link:\n"
        "📦 Buy Online Link:\n"
        "🔄 Ask Again:"
    )
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}},
        ],
        stream=False,
    )

    cleaned = remove_duplicates(response.text)
    uploaded_image_url = f"/{path.replace(os.sep, '/')}"

    save_message("model", cleaned, source="image")

    return jsonify({"result": cleaned, "uploaded_image": uploaded_image_url})


if __name__ == "__main__":
    app.run(debug=True)
