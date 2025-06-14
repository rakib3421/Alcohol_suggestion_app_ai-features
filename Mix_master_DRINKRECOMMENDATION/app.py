import os
import requests
import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env
load_dotenv()

# Flask setup
app = Flask(__name__)

# API Keys
WEATHER_API_KEY = os.getenv("OPEN_WEATHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini setup
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")


def get_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    print("📡 Requesting:", url)
    res = requests.get(url)
    data = res.json()
    print("📦 Response JSON:", data)

    if res.status_code != 200 or "weather" not in data or "main" not in data:
        return None

    weather = data["weather"][0]["description"].capitalize()
    temp = data["main"]["temp"]
    country = data["sys"]["country"]
    return {"summary": f"{weather}, {temp}°C", "country": country}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_suggestion", methods=["POST"])
def get_suggestion():
    try:
        lat = request.json.get("lat")
        lon = request.json.get("lon")
        mood = request.json.get("mood")

        weather_data = get_weather(lat, lon)
        if not weather_data:
            print("❌ Weather API returned None or error")
            return jsonify({"error": "Weather data unavailable"}), 400

        weather_info = weather_data["summary"]
        country = weather_data["country"]

        hour = datetime.datetime.now().hour
        time_of_day = (
            "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
        )

        prompt = f"""
        You are an expert alcohol and food pairing assistant.

        Based on the following:
        - Current weather: {weather_info}
        - Time of day: {time_of_day}
        - Mood: {mood}
        - Country: {country}

        Suggest the following:
        1. A **specific alcohol brand** (preferably one available/popular in {country}).
        2. A **recommended food item** that complements it well, based on the weather and mood.
        3. A short, fun reason why this pairing is perfect right now.

        Make the response short, local-friendly, and under 3 sentences. Keep it engaging and relevant to the local vibe.
        """
        response = model.generate_content(prompt)
        suggestion = response.text.strip()
        return jsonify({"suggestion": suggestion})
    except Exception as e:
        print("🔥 Error:", e)
        return jsonify({"error": "Internal Server Error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
