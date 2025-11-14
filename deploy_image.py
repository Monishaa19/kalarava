import os
from io import BytesIO
from dotenv import load_dotenv
from google import genai
from google.genai import types
from flask import Flask, send_file
import time
import logging

load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set. Check your .env file.")

client = genai.Client(api_key=api_key)

@app.route("/", methods=["GET"])
def index():
    return {
        "message": "Welcome to Image Generation API",
        "usage": "/generate-image/<description>",
        "example": "/generate-image/a%20beautiful%20landscape"
    }, 200

@app.route("/generate-image/<description>", methods=["GET"])
def generate_image(description):
    try:
        if not description or len(description.strip()) == 0:
            return {"error": "Description cannot be empty"}, 400
        
        logger.info(f"Generating image for: {description}")
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
                        image_bytes = BytesIO(part.inline_data.data)
                        logger.info("Image generated successfully")
                        return send_file(image_bytes, mimetype="image/png"), 200
                
                break
                
            except Exception as e:
                retry_count += 1
                if "429" in str(e) and retry_count < max_retries:
                    wait_time = 2 ** retry_count
                    logger.warning(f"Rate limited. Retrying in {wait_time}s... (Attempt {retry_count}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error generating image: {e}")
                    raise
        
        return {"error": "No image generated"}, 500
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}, 500

@app.errorhandler(404)
def not_found(error):
    return {"error": "Endpoint not found"}, 404

@app.errorhandler(500)
def server_error(error):
    return {"error": "Internal server error"}, 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)