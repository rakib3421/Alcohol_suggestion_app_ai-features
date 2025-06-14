import base64
import os
import json
import re
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import mysql.connector
import openai
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import requests

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-here")


# Configuration
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MYSQL_HOST = os.getenv("MYSQL_HOST")
    MYSQL_USER = os.getenv("MYSQL_USER")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
    MYSQL_DB = os.getenv("MYSQL_DB")
    UPLOAD_FOLDER = "static/uploads"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}


# Initialize OpenAI
openai.api_key = Config.OPENAI_API_KEY

# Database configuration
db_config = (
    {
        "host": Config.MYSQL_HOST,
        "user": Config.MYSQL_USER,
        "password": Config.MYSQL_PASSWORD,
        "database": Config.MYSQL_DB,
    }
    if all(
        [Config.MYSQL_HOST, Config.MYSQL_USER, Config.MYSQL_PASSWORD, Config.MYSQL_DB]
    )
    else None
)

# Create upload directory
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = Config.UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )


def optimize_image(image_path: str, max_size: tuple = (1024, 1024)) -> None:
    """Optimize image size and quality for API processing."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Resize if too large
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Save optimized image
            img.save(image_path, "JPEG", quality=85, optimize=True)
            logger.info(f"Image optimized: {image_path}")
    except Exception as e:
        logger.error(f"Error optimizing image {image_path}: {str(e)}")


def get_alcohol_info_from_ai(
    image_path: str, extra_ingredients: str = "", gastronomy: str = ""
) -> Dict[str, Any]:
    """Get alcohol information from OpenAI Vision API."""
    try:
        # Read and encode image
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Construct prompt with additional context
        prompt = f"""
Analyze the uploaded image of alcohol and provide a detailed structured JSON object with the following keys:

Required fields:
- brand (string): The brand name of the alcohol
- origin (string): Country/region of origin
- alcohol_content (string): ABV percentage or strength
- ingredients (array of strings): Main ingredients used in production
- tasting_notes (string): Detailed flavor profile and characteristics
- similar_alcohols (array of strings): 3-5 similar products
- recipes (array of objects): 2-3 cocktail recipes with this alcohol, each containing:
  - name (string): Recipe name
  - description (string): Brief description
  - ingredients (array of strings): List of ingredients with measurements
  - preparation (array of strings): Step-by-step instructions
- confidence_level (number): Your confidence in the identification (0-100)
- brand_info (string): Brief history and information about the brand
- youtube_links (array of strings): 2-3 relevant YouTube video URLs for cocktail tutorials
- purchase_options (array of strings): 2-3 online store URLs where it can be purchased
- brand_logo (string): URL to brand logo image if available

Additional context:
- Extra ingredients requested: {extra_ingredients if extra_ingredients else "None"}
- Gastronomy preferences: {gastronomy if gastronomy else "None"}

If extra ingredients are provided, incorporate them into the recipes.
If gastronomy preferences are mentioned, suggest food pairings in the brand_info section.

Return ONLY valid JSON, no markdown formatting or additional text.
Ensure all URLs are realistic and functional.
"""

        # Make API call
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert sommelier and mixologist with extensive knowledge of alcoholic beverages, their production, history, and cocktail preparation. Provide accurate, detailed information.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                },
            ],
            max_tokens=2000,
            temperature=0.7,
        )

        json_response = response.choices[0].message.content.strip()
        logger.info("Received response from OpenAI API")

        # Clean and parse JSON response
        json_str_match = re.search(r"\{.*\}", json_response, re.DOTALL)
        if json_str_match:
            json_str = json_str_match.group()
            data = json.loads(json_str)

            # Validate required fields
            required_fields = [
                "brand",
                "origin",
                "alcohol_content",
                "ingredients",
                "tasting_notes",
            ]
            for field in required_fields:
                if field not in data:
                    data[field] = "Information not available"

            # Ensure arrays exist
            for field in [
                "ingredients",
                "similar_alcohols",
                "recipes",
                "youtube_links",
                "purchase_options",
            ]:
                if field not in data:
                    data[field] = []

            # Ensure confidence level is a number
            if "confidence_level" not in data:
                data["confidence_level"] = 75

            logger.info(
                f"Successfully processed alcohol information for: {data.get('brand', 'Unknown')}"
            )
            return data
        else:
            raise ValueError("No valid JSON found in response")

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        return {
            "error": f"Failed to parse AI response as JSON: {str(e)}",
            "raw_response": (
                json_response if "json_response" in locals() else "No response received"
            ),
        }
    except Exception as e:
        logger.error(f"Error in AI processing: {str(e)}")
        return {"error": f"Error processing image: {str(e)}", "raw_response": ""}


def save_to_database(
    filename: str, ingredients: str, gastronomy: str, ai_response: str
) -> bool:
    """Save analysis results to database."""
    if not db_config:
        logger.warning("Database configuration not available")
        return False

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alcohol_analyses (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                ingredients TEXT,
                gastronomy TEXT,
                ai_response JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Insert record
        cursor.execute(
            """
            INSERT INTO alcohol_analyses (filename, ingredients, gastronomy, ai_response)
            VALUES (%s, %s, %s, %s)
        """,
            (filename, ingredients, gastronomy, ai_response),
        )

        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"Successfully saved analysis to database for file: {filename}")
        return True

    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        return False


@app.route("/", methods=["GET", "POST"])
def index():
    """Main route for image upload and analysis."""
    if request.method == "POST":
        try:
            # Validate file upload
            if "image" not in request.files:
                flash("No file selected", "error")
                return redirect(request.url)

            image = request.files["image"]
            if image.filename == "":
                flash("No file selected", "error")
                return redirect(request.url)

            if not allowed_file(image.filename):
                flash(
                    "Invalid file type. Please upload PNG, JPG, JPEG, GIF, or WebP images.",
                    "error",
                )
                return redirect(request.url)

            # Get additional inputs
            ingredients = request.form.get("ingredients", "").strip()
            gastronomy = request.form.get("gastronomy", "").strip()

            # Save uploaded file
            filename = secure_filename(image.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
            filename = timestamp + filename
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image.save(image_path)

            logger.info(f"Image uploaded: {filename}")

            # Optimize image for API processing
            optimize_image(image_path)

            # Get AI analysis
            data = get_alcohol_info_from_ai(image_path, ingredients, gastronomy)

            # Save to database if successful
            if "error" not in data:
                ai_response_json = json.dumps(data)
                save_to_database(filename, ingredients, gastronomy, ai_response_json)
                flash("Analysis completed successfully!", "success")
            else:
                flash(
                    "Analysis completed with some issues. Please check the results.",
                    "warning",
                )

            return render_template("index.html", data=data)

        except Exception as e:
            logger.error(f"Error in main route: {str(e)}")
            flash(f"An error occurred during processing: {str(e)}", "error")
            return redirect(request.url)

    return render_template("index.html", data=None)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """API endpoint for image analysis (JSON response)."""
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image = request.files["image"]
        if not allowed_file(image.filename):
            return jsonify({"error": "Invalid file type"}), 400

        # Save and process image
        filename = secure_filename(image.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image.save(image_path)

        optimize_image(image_path)

        # Get additional parameters
        ingredients = request.form.get("ingredients", "")
        gastronomy = request.form.get("gastronomy", "")

        # Analyze with AI
        data = get_alcohol_info_from_ai(image_path, ingredients, gastronomy)

        # Save to database
        if "error" not in data:
            ai_response_json = json.dumps(data)
            save_to_database(filename, ingredients, gastronomy, ai_response_json)

        return jsonify(data)

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/history")
def history():
    """View analysis history."""
    if not db_config:
        flash("Database not configured", "error")
        return redirect(url_for("index"))

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT id, filename, ingredients, gastronomy, ai_response, created_at
            FROM alcohol_analyses
            ORDER BY created_at DESC
            LIMIT 50
        """
        )

        analyses = cursor.fetchall()

        # Parse JSON responses
        for analysis in analyses:
            if analysis["ai_response"]:
                try:
                    analysis["parsed_response"] = json.loads(analysis["ai_response"])
                except:
                    analysis["parsed_response"] = None

        cursor.close()
        conn.close()

        return render_template("history.html", analyses=analyses)

    except Exception as e:
        logger.error(f"History error: {str(e)}")
        flash(f"Error loading history: {str(e)}", "error")
        return redirect(url_for("index"))


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash("File is too large. Maximum size is 16MB.", "error")
    return redirect(request.url), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(error)}")
    flash("An internal error occurred. Please try again.", "error")
    return redirect(url_for("index")), 500


if __name__ == "__main__":
    # Check for required environment variables
    if not Config.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not found in environment variables")
        print("Please set OPENAI_API_KEY in your .env file")
        exit(1)

    logger.info("Starting Alcohol Recipe & Info Generator")
    app.run(debug=True, host="0.0.0.0", port=5000)
