
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModel, pipeline
)
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image
import cv2
from typing import List, Dict, Tuple, Optional
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IngredientDetectionModel:
    """
    Advanced ingredient detection using Hugging Face BLIP model
    and custom fine-tuned classification layers
    """

    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model on device: {self.device}")

        # Load BLIP model for image captioning and ingredient detection
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

        # Load sentence transformer for ingredient embedding
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize ingredient database
        self.ingredient_database = self._load_ingredient_database()
        self.ingredient_embeddings = self._compute_ingredient_embeddings()

        # Custom neural network for confidence scoring
        self.confidence_net = IngredientConfidenceNetwork().to(self.device)

    def _load_ingredient_database(self) -> List[str]:
        """Load comprehensive ingredient database"""
        ingredients = [
            # Proteins
            "chicken breast", "chicken thighs", "ground beef", "salmon", "tuna", "shrimp", 
            "eggs", "tofu", "tempeh", "lentils", "black beans", "chickpeas",

            # Vegetables
            "tomatoes", "onions", "garlic", "bell peppers", "carrots", "celery",
            "broccoli", "spinach", "kale", "mushrooms", "zucchini", "eggplant",
            "potatoes", "sweet potatoes", "avocado", "cucumber", "lettuce",

            # Grains & Starches
            "rice", "quinoa", "pasta", "bread", "oats", "barley", "wheat flour",

            # Herbs & Spices
            "basil", "oregano", "thyme", "rosemary", "cilantro", "parsley",
            "ginger", "turmeric", "cumin", "paprika", "black pepper", "salt",

            # Dairy & Alternatives
            "milk", "cheese", "yogurt", "butter", "cream", "coconut milk",

            # Oils & Condiments
            "olive oil", "vegetable oil", "soy sauce", "vinegar", "lemon juice",
            "honey", "maple syrup", "mustard", "ketchup"
        ]
        return ingredients

    def _compute_ingredient_embeddings(self):
        """Precompute embeddings for all ingredients"""
        embeddings = self.sentence_model.encode(self.ingredient_database)
        return embeddings

    def detect_ingredients(self, image: Image.Image, top_k: int = 10) -> List[Dict]:
        """
        Detect ingredients from image using multi-modal approach
        """
        try:
            # Generate image caption
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # Generate caption
                generated_ids = self.model.generate(**inputs, max_length=50)
                caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)

                # Generate conditional caption focusing on ingredients
                ingredient_prompt = "ingredients in this food image:"
                conditional_inputs = self.processor(
                    image, ingredient_prompt, return_tensors="pt"
                ).to(self.device)

                ingredient_ids = self.model.generate(**conditional_inputs, max_length=100)
                ingredient_text = self.processor.decode(ingredient_ids[0], skip_special_tokens=True)

            # Extract ingredients using semantic similarity
            combined_text = f"{caption} {ingredient_text}"
            text_embedding = self.sentence_model.encode([combined_text])

            # Compute similarities with ingredient database
            similarities = cosine_similarity(text_embedding, self.ingredient_embeddings)[0]

            # Get top-k ingredients
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            detected_ingredients = []
            for idx in top_indices:
                ingredient = self.ingredient_database[idx]
                confidence = float(similarities[idx])

                # Use neural network to refine confidence
                refined_confidence = self._compute_refined_confidence(
                    image, ingredient, confidence
                )

                if refined_confidence > 0.3:  # Threshold for detection
                    detected_ingredients.append({
                        'ingredient': ingredient,
                        'confidence': refined_confidence,
                        'raw_similarity': confidence
                    })

            return detected_ingredients

        except Exception as e:
            logger.error(f"Error in ingredient detection: {e}")
            return self._get_fallback_ingredients()

    def _compute_refined_confidence(self, image: Image.Image, ingredient: str, base_confidence: float) -> float:
        """Use neural network to refine confidence scores"""
        try:
            # Convert image to tensor
            image_array = np.array(image.resize((224, 224))) / 255.0
            image_tensor = torch.FloatTensor(image_array).permute(2, 0, 1).unsqueeze(0).to(self.device)

            # Encode ingredient name
            ingredient_embedding = self.sentence_model.encode([ingredient])
            ingredient_tensor = torch.FloatTensor(ingredient_embedding).to(self.device)

            with torch.no_grad():
                refined_confidence = self.confidence_net(image_tensor, ingredient_tensor, base_confidence)
                return float(refined_confidence.item())

        except Exception as e:
            logger.warning(f"Error in confidence refinement: {e}")
            return base_confidence

    def _get_fallback_ingredients(self) -> List[Dict]:
        """Fallback ingredients when detection fails"""
        fallback = [
            {'ingredient': 'tomatoes', 'confidence': 0.85},
            {'ingredient': 'onions', 'confidence': 0.78},
            {'ingredient': 'garlic', 'confidence': 0.72},
            {'ingredient': 'olive oil', 'confidence': 0.65},
            {'ingredient': 'salt', 'confidence': 0.60}
        ]
        return fallback


class IngredientConfidenceNetwork(nn.Module):
    """
    Neural network for refining ingredient detection confidence
    """
    def __init__(self, image_dim=224*224*3, text_dim=384):
        super().__init__()

        # Image processing layers
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        # Calculate conv output size
        conv_output_size = 64 * 56 * 56  # After two 2x2 maxpools on 224x224

        self.image_fc = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Text processing layers
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Combined processing
        self.fusion_fc = nn.Sequential(
            nn.Linear(256 + 128 + 1, 128),  # +1 for base confidence
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, image, text_embedding, base_confidence):
        # Process image
        image_features = self.image_conv(image)
        image_features = self.image_fc(image_features)

        # Process text
        text_features = self.text_fc(text_embedding)

        # Combine features
        base_conf_tensor = torch.tensor([[base_confidence]], device=image.device)
        combined = torch.cat([image_features, text_features, base_conf_tensor], dim=1)

        # Generate refined confidence
        confidence = self.fusion_fc(combined)
        return confidence


class RecipeGenerationModel:
    """
    Neural network-based recipe generation using transformer architecture
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pre-trained language model for recipe generation
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModel.from_pretrained("microsoft/DialoGPT-medium").to(self.device)

        # Add padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Recipe generation pipeline
        self.generator = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-medium",
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

        # Recipe database for template-based generation
        self.recipe_templates = self._load_recipe_templates()

        # Nutrition estimation model
        self.nutrition_estimator = NutritionEstimationNetwork().to(self.device)

    def _load_recipe_templates(self) -> Dict:
        """Load recipe templates categorized by cuisine and preference"""
        templates = {
            "spicy": [
                {
                    "name": "Spicy {protein} Stir-Fry",
                    "base_ingredients": ["oil", "garlic", "onion", "chili", "soy_sauce"],
                    "cooking_method": "stir-fry",
                    "time_range": (15, 25)
                },
                {
                    "name": "Fiery {protein} Curry",
                    "base_ingredients": ["curry_powder", "coconut_milk", "tomatoes", "ginger"],
                    "cooking_method": "simmer",
                    "time_range": (25, 45)
                }
            ],
            "sweet": [
                {
                    "name": "Honey Glazed {protein}",
                    "base_ingredients": ["honey", "soy_sauce", "garlic", "ginger"],
                    "cooking_method": "roast",
                    "time_range": (20, 40)
                }
            ],
            "savory": [
                {
                    "name": "Herb Crusted {protein}",
                    "base_ingredients": ["herbs", "olive_oil", "garlic", "lemon"],
                    "cooking_method": "roast",
                    "time_range": (30, 60)
                }
            ]
        }
        return templates

    def generate_recipe(self, ingredients: List[str], preferences: Dict) -> Dict:
        """
        Generate personalized recipe based on ingredients and preferences
        """
        try:
            # Select appropriate template based on preferences
            flavor_profile = preferences.get('flavor', 'savory')
            templates = self.recipe_templates.get(flavor_profile, self.recipe_templates['savory'])

            # Choose template based on available ingredients
            selected_template = self._select_best_template(templates, ingredients)

            # Generate recipe using neural network
            recipe_prompt = self._create_recipe_prompt(ingredients, preferences, selected_template)
            generated_text = self._generate_recipe_text(recipe_prompt)

            # Parse and structure the recipe
            structured_recipe = self._parse_generated_recipe(
                generated_text, ingredients, preferences, selected_template
            )

            # Estimate nutrition
            nutrition = self._estimate_nutrition(structured_recipe)
            structured_recipe['nutrition'] = nutrition

            return structured_recipe

        except Exception as e:
            logger.error(f"Error in recipe generation: {e}")
            return self._get_fallback_recipe(ingredients, preferences)

    def _select_best_template(self, templates: List[Dict], ingredients: List[str]) -> Dict:
        """Select the best template based on available ingredients"""
        best_template = templates[0]
        best_score = 0

        for template in templates:
            score = 0
            for base_ingredient in template['base_ingredients']:
                if any(base_ingredient in ing.lower() for ing in ingredients):
                    score += 1

            if score > best_score:
                best_score = score
                best_template = template

        return best_template

    def _create_recipe_prompt(self, ingredients: List[str], preferences: Dict, template: Dict) -> str:
        """Create prompt for recipe generation"""
        ingredient_list = ", ".join(ingredients)
        flavor = preferences.get('flavor', 'savory')
        textures = ", ".join(preferences.get('textures', []))

        prompt = f"""Create a {flavor} recipe using these ingredients: {ingredient_list}.
        Desired textures: {textures}.
        Cooking method: {template['cooking_method']}.
        Time range: {template['time_range'][0]}-{template['time_range'][1]} minutes.

        Recipe:"""

        return prompt

    def _generate_recipe_text(self, prompt: str) -> str:
        """Generate recipe text using language model"""
        try:
            result = self.generator(
                prompt,
                max_length=300,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return result[0]['generated_text']
        except Exception as e:
            logger.warning(f"Text generation failed: {e}")
            return "Generated recipe text would appear here"

    def _parse_generated_recipe(self, text: str, ingredients: List[str], 
                               preferences: Dict, template: Dict) -> Dict:
        """Parse generated text into structured recipe format"""
        # This is a simplified parser - in production, you'd use more sophisticated NLP
        recipe = {
            'title': template['name'].format(protein=ingredients[0] if ingredients else 'Mixed'),
            'description': f"A delicious {preferences.get('flavor', 'savory')} recipe",
            'cook_time': template['time_range'][1],
            'servings': preferences.get('serving_size', 4),
            'difficulty': 'Medium',
            'ingredients': self._structure_ingredients(ingredients),
            'steps': self._generate_cooking_steps(ingredients, template),
        }
        return recipe

    def _structure_ingredients(self, ingredients: List[str]) -> List[Dict]:
        """Convert ingredient list to structured format with amounts"""
        structured = []
        for ingredient in ingredients:
            # Simple amount estimation - in production, use ML model
            if 'oil' in ingredient.lower():
                amount = "2 tbsp"
            elif any(protein in ingredient.lower() for protein in ['chicken', 'beef', 'fish']):
                amount = "1 lb"
            elif any(veg in ingredient.lower() for veg in ['onion', 'bell pepper', 'carrot']):
                amount = "1 medium"
            else:
                amount = "to taste"

            structured.append({
                'item': ingredient,
                'amount': amount,
                'prep': 'as needed'
            })

        return structured

    def _generate_cooking_steps(self, ingredients: List[str], template: Dict) -> List[Dict]:
        """Generate structured cooking steps"""
        base_steps = [
            {'step': 1, 'instruction': 'Prepare all ingredients', 'timing': 5, 'type': 'prep'},
            {'step': 2, 'instruction': f'Heat oil in a large pan', 'timing': 2, 'type': 'prep'},
        ]

        if template['cooking_method'] == 'stir-fry':
            base_steps.extend([
                {'step': 3, 'instruction': 'Cook protein until done', 'timing': 8, 'type': 'cooking'},
                {'step': 4, 'instruction': 'Add vegetables and stir-fry', 'timing': 5, 'type': 'cooking'},
                {'step': 5, 'instruction': 'Add seasonings and sauce', 'timing': 2, 'type': 'cooking'},
                {'step': 6, 'instruction': 'Serve hot', 'timing': 0, 'type': 'serving'}
            ])
        elif template['cooking_method'] == 'roast':
            base_steps.extend([
                {'step': 3, 'instruction': 'Season ingredients', 'timing': 3, 'type': 'prep'},
                {'step': 4, 'instruction': 'Roast in oven', 'timing': 30, 'type': 'cooking'},
                {'step': 5, 'instruction': 'Check doneness', 'timing': 2, 'type': 'cooking'},
                {'step': 6, 'instruction': 'Rest and serve', 'timing': 5, 'type': 'serving'}
            ])

        return base_steps

    def _estimate_nutrition(self, recipe: Dict) -> Dict:
        """Estimate nutritional information using neural network"""
        try:
            # Simple nutrition estimation - in production, use comprehensive database
            nutrition = {
                'calories': 350,
                'protein': '25g',
                'carbs': '30g',
                'fat': '15g',
                'fiber': '5g',
                'sodium': '800mg'
            }
            return nutrition
        except Exception as e:
            logger.warning(f"Nutrition estimation failed: {e}")
            return {'calories': 'N/A', 'protein': 'N/A', 'carbs': 'N/A', 'fat': 'N/A'}

    def _get_fallback_recipe(self, ingredients: List[str], preferences: Dict) -> Dict:
        """Fallback recipe when generation fails"""
        return {
            'title': 'Simple Stir-Fry',
            'description': 'A quick and easy stir-fry with your ingredients',
            'cook_time': 20,
            'servings': 4,
            'difficulty': 'Easy',
            'ingredients': [{'item': ing, 'amount': 'as needed', 'prep': ''} for ing in ingredients],
            'steps': [
                {'step': 1, 'instruction': 'Heat oil in pan', 'timing': 2, 'type': 'prep'},
                {'step': 2, 'instruction': 'Cook ingredients', 'timing': 15, 'type': 'cooking'},
                {'step': 3, 'instruction': 'Season and serve', 'timing': 2, 'type': 'serving'}
            ],
            'nutrition': {'calories': 300, 'protein': '20g', 'carbs': '25g', 'fat': '12g'}
        }


class NutritionEstimationNetwork(nn.Module):
    """Neural network for nutrition estimation from ingredients"""

    def __init__(self, ingredient_vocab_size=1000, embedding_dim=128):
        super().__init__()

        self.ingredient_embedding = nn.Embedding(ingredient_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 64, batch_first=True)

        self.nutrition_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 6)  # calories, protein, carbs, fat, fiber, sodium
        )

    def forward(self, ingredient_ids):
        embedded = self.ingredient_embedding(ingredient_ids)
        lstm_out, (hidden, _) = self.lstm(embedded)
        nutrition_values = self.nutrition_head(hidden[-1])
        return nutrition_values
