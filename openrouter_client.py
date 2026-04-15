import requests
import base64
import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode('utf-8')

def analyze_image_for_ingredients(image_bytes: bytes, api_key: str = "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", model: str = "google/gemini-2.0-flash-001") -> List[Dict]:
    """
    Use an OpenRouter VLM to identify ingredients in an image.
    Returns a list of dictionaries with 'ingredient' and 'confidence'.
    """
    if not api_key:
        raise ValueError("OpenRouter API key is required")

    base64_image = encode_image_to_base64(image_bytes)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501", # Optional, for OpenRouter analytics
        "X-Title": "AI Recipe Generator" # Optional, for OpenRouter analytics
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Identify all the food ingredients visible in this image. Return the output STRICTLY as a JSON array of objects, where each object has an 'ingredient' field (string, name of ingredient) and a 'confidence' field (number between 0.0 and 1.0, your confidence level). Do not return any other text, markdown blocks, or explanation, just the raw JSON array."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        content = data['choices'][0]['message']['content'].strip()
        
        # Clean up markdown JSON formatting if present
        if content.startswith('```json'):
            content = content[7:-3]
        elif content.startswith('```'):
            content = content[3:-3]
            
        ingredients = json.loads(content)
        
        results = []
        for item in ingredients:
            results.append({
                'ingredient': str(item.get('ingredient', 'Unknown')),
                'confidence': float(item.get('confidence', 0.8)),
                'source': 'OpenRouter API',
                'reasoning': 'Detected via Large Vision Model'
            })
            
        return results
    except Exception as e:
        logger.error(f"OpenRouter image analysis failed: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            logger.error(f"Response: {response.text}")
        raise

def generate_recipe(ingredients: List[str], preferences: Dict, api_key: str = "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", model: str = "google/gemini-2.0-flash-001") -> Dict:
    """
    Use an OpenRouter LLM to generate a recipe based on ingredients and preferences.
    """
    if not api_key:
        raise ValueError("OpenRouter API key is required")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "AI Recipe Generator"
    }

    prefs_str = json.dumps(preferences, indent=2)
    ingredients_str = ", ".join(ingredients)

    prompt = f"""
ACT AS AN EXPERT CHEF. Create a highly authentic recipe based on these parameters:
INGREDIENTS: {ingredients_str}
USER_PREFERENCES_JSON: 
{prefs_str}

CONSTRAINTS:
1. Stay strictly within the requested 'cooking_time' and 'difficulty'.
2. Honor the multi-sensory profile (Flavor Palette & Texture intensities).
3. Ensure the recipe matches the 'cuisine' selected for authentic flavor profiles.
4. Every cooking step with a duration MUST have a 'timing' value in minutes.

RETURN ONLY RAW JSON (NO MARKDOWN):
{{
  "title": "...",
  "description": "...",
  "cook_time": 0,
  "servings": 0,
  "difficulty": "...",
  "sensory_profile": {{ "taste": "...", "texture": "..." }},
  "ingredients": [{{ "item": "...", "amount": "...", "prep": "..." }}],
  "steps": [{{ "step": 1, "instruction": "...", "timing": 5, "type": "cooking/prep" }}]
}}
"""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        content = data['choices'][0]['message']['content'].strip()
        
        # Clean up markdown JSON formatting if present
        if content.startswith('```json'):
            content = content[7:-3]
        elif content.startswith('```'):
            content = content[3:-3]
            
        recipe = json.loads(content)
        return recipe
    except Exception as e:
        logger.error(f"OpenRouter recipe generation failed: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            logger.error(f"Response: {response.text}")
        raise
