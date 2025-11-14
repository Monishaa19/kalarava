import os
from io import BytesIO
from dotenv import load_dotenv
from google import genai
from google.genai import types
from flask import Flask, send_file
from flask import jsonify
import time
import logging

load_dotenv()

app = Flask(__name__)

api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    # In production, it's better to log this than to crash on startup
    # The service will fail health checks if the key is missing.
    print("Warning: GEMINI_API_KEY environment variable not set.")

client = genai.Client(api_key=api_key)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route("/generate-image/<description>", methods=["GET"])
def generate_image(description):
    try:
        if not description or len(description.strip()) == 0:
            return jsonify({"error": "Description cannot be empty"}), 400

        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash-preview-image-generation",
                    contents=description,
                    config=types.GenerateContentConfig(response_modalities=['TEXT', 'IMAGE'])
                )
                
                # Extract image
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        logger.info(f"Image generated successfully for: {description}")
                        image_bytes = BytesIO(part.inline_data.data)
                        return send_file(image_bytes, mimetype="image/png"), 200
                
                break
                
            except Exception as e:
                retry_count += 1
                if "429" in str(e) and retry_count < max_retries:
                    wait_time = 2 ** retry_count
                    logger.warning(f"Rate limited. Retrying in {wait_time}s... (Attempt {retry_count}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error generating image after {retry_count} retries: {e}")
                    raise
        
        logger.error("No image generated after retries.")
        return jsonify({"error": "No image generated"}), 500
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return jsonify({
        "message": "Welcome to the Image Generation API",
        "usage": "/generate-image/<description>",
        "example": "/generate-image/a%20beautiful%20painting%20of%20a%20sunset"
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # This block is for local development, Render will use the Gunicorn start command.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
