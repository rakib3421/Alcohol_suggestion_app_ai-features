from flask import Flask, request, jsonify, send_from_directory
import os
import json
import uuid
import google.generativeai as genai
from dotenv import load_dotenv
import re

# Load API key from .env
load_dotenv()
genai_api_key = os.getenv("GEMINI_API_KEY")
if not genai_api_key:
    raise EnvironmentError("❌ GEMINI_API_KEY not found in .env file")

# Configure Gemini API
genai.configure(api_key=genai_api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def extract_json(text: str):
    """Try to extract JSON object from text using regex."""
    json_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not json_match:
        return None
    try:
        return json.loads(json_match.group(1))
    except json.JSONDecodeError:
        return None


def generate_recipe(image_url: str, description: str) -> dict | None:
    """Generate cocktail recipe JSON by sending image URL and description to Gemini."""
    try:
        prompt = f"""
You are a professional mixologist and beverage expert.

Analyze the alcohol bottle shown in this image: {image_url}

Use the following user description for extra context: "{description}".

Based on this, generate a complete structured JSON object that includes accurate cocktail metadata, flavor insights, mixology steps, and pairing suggestions.

⚠️ Output ONLY valid JSON. Do NOT include explanations, markdown, or any extra text.

The JSON schema must exactly match this structure:
{{
  "name": "name of the alochol",
  "alcohol_content": "String",
  "type": "String",
  "description": "String",
  "image": "String",
  "flavor_profile": "String",
  "strength": "String",
  "difficulty": "String",
  "glass": "String",
  "rating": {{
    "score": Float,
    "total_ratings": Integer
  }},
  "tags": ["String"],
  "ingredients": [
    {{
      "name": "String",
      "amount": "String",
      "category": "String"
    }}
  ],
  "garnish": ["String"],
  "instructions": {{
    "how_to_make": "String",
    "steps": [
      {{
        "step": Integer,
        "title": "String",
        "instruction": "String",
        "tip": "String"
      }}
    ]
  }},
  "variations": [
    {{
      "name": "String",
      "description": "String",
      "key_ingredient": "String"
    }}
  ],
  "serving_info": {{
    "best_time": "String",
    "occasion": "String",
    "temperature": "String",
    "garnish_placement": "String"
  }},
  "nutritional_info": {{
    "calories": Integer,
    "alcohol_content": "String",
    "sugar_content": "String"
  }},
  "pairing_recommendations": [
    {{
      "category": "String",
      "items": ["String"],
      "emoji": "String"
    }}
  ],
  "professional_tips": ["String"],
  "history": {{
    "origin": "String",
    "creator": "String",
    "year_created": "String",
    "story": "String"
  }}
}}
"""
        response = model.generate_content([prompt])
        raw_text = response.text.strip()
        data = extract_json(raw_text)
        if not data:
            print("❌ Failed to parse JSON from Gemini response.")
            print(f"Raw response:\n{raw_text}")
        return data

    except Exception as e:
        print(f"❌ Exception in generate_recipe(): {e}")
        return None


@app.route("/generate-cocktail", methods=["POST"])
def cocktail_api():
    if "image" not in request.files or "description" not in request.form:
        return jsonify({"error": "Missing image or description"}), 400

    image_file = request.files["image"]
    description = request.form["description"]

    try:
        # Save uploaded image with unique filename
        filename = f"{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image_file.save(save_path)

        # Construct image URL accessible by Gemini (must be public or tunneled)
        image_url = request.host_url + f"uploads/{filename}"

        # Generate cocktail data from image URL and description
        data = generate_recipe(image_url, description)
        if not data:
            return jsonify({"error": "Failed to generate cocktail data"}), 500

        return jsonify(data), 200

    except Exception as e:
        print(f"❌ Error in /generate-cocktail route: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🍸 Welcome to the Cocktail Recipe Generator API"})


if __name__ == "__main__":
    app.run(debug=True)
