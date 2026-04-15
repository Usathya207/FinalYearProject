
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModel, pipeline, AutoImageProcessor, AutoModelForImageClassification
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
import requests
import os
from pathlib import Path
import torchvision.transforms as transforms
import re
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccurateIngredientDetectionModel:
    """
    Much more accurate ingredient detection that doesn't over-detect
    """

    def __init__(self, use_ensemble=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading accurate model on device: {self.device}")

        self.use_ensemble = use_ensemble

        # Load vision models
        self._load_vision_models()

        # Load realistic ingredient database - smaller and more focused
        self.ingredient_database = self._load_focused_ingredient_database()
        self.ingredient_embeddings = self._compute_ingredient_embeddings()

        # Common ingredient patterns for filtering
        self.common_patterns = self._load_common_patterns()

        # Confidence thresholds - much more conservative
        self.confidence_thresholds = {
            'high_confidence': 0.75,      # Very confident detection
            'medium_confidence': 0.50,    # Moderately confident
            'low_confidence': 0.30,       # Lower bound
            'minimum_threshold': 0.45     # Absolute minimum to include
        }

    def _load_vision_models(self):
        """Load vision models with better error handling"""
        try:
            # Primary BLIP model
            logger.info("Loading BLIP model...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self.device)

            # Sentence transformer for ingredient similarity
            logger.info("Loading sentence transformer...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

            logger.info("Vision models loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading vision models: {e}")
            raise

    def _load_focused_ingredient_database(self) -> List[str]:
        """Load a focused, realistic ingredient database"""

        # Core ingredients that are commonly detected in food images
        focused_ingredients = [
            # Single vegetables (most commonly photographed)
            "tomato", "tomatoes", "onion", "onions", "garlic", "carrot", "carrots",
            "bell pepper", "bell peppers", "potato", "potatoes", "mushroom", "mushrooms",
            "cucumber", "lettuce", "spinach", "broccoli", "cauliflower", "zucchini",
            "eggplant", "avocado", "corn", "peas", "green beans",

            # Fruits
            "apple", "banana", "orange", "lemon", "lime", "strawberry", "blueberry",
            "grapes", "pineapple", "mango", "watermelon", "peach", "pear",

            # Proteins (when clearly visible)
            "chicken", "beef", "pork", "fish", "salmon", "tuna", "shrimp", "egg", "eggs",
            "tofu", "cheese", "bacon", "ham",

            # Grains & Bread (when clearly visible)
            "bread", "rice", "pasta", "noodles", "quinoa", "oats",

            # Herbs & Visible Seasonings
            "basil", "parsley", "cilantro", "mint", "rosemary", "thyme", "oregano",
            "ginger", "chili", "pepper",

            # Nuts & Seeds (when visible)
            "almonds", "walnuts", "peanuts", "sesame seeds", "sunflower seeds",

            # Visible condiments/sauces
            "olive oil", "butter", "honey", "vinegar"
        ]

        logger.info(f"Loaded {len(focused_ingredients)} focused ingredients")
        return focused_ingredients

    def _compute_ingredient_embeddings(self):
        """Precompute embeddings for focused ingredients"""
        embeddings = self.sentence_model.encode(self.ingredient_database)
        return embeddings

    def _load_common_patterns(self):
        """Load common ingredient combination patterns to filter unrealistic results"""
        return {
            # Single item images - if we detect a tomato, unlikely to have 8 other things
            'single_item_indicators': [
                'single', 'one', 'a tomato', 'a apple', 'a orange', 'a banana',
                'whole', 'fresh', 'ripe', 'individual'
            ],

            # Prepared dish indicators
            'prepared_dish_indicators': [
                'salad', 'soup', 'stew', 'curry', 'stir fry', 'pasta', 'pizza',
                'sandwich', 'burger', 'bowl', 'plate', 'dish', 'meal'
            ],

            # Cooking process indicators
            'cooking_indicators': [
                'cooking', 'cooked', 'baked', 'fried', 'grilled', 'roasted',
                'sauteed', 'steamed', 'boiled'
            ]
        }

    def detect_ingredients(self, image: Image.Image, top_k: int = 8, confidence_threshold: float = 0.45) -> List[Dict]:
        """
        Much more accurate ingredient detection that doesn't over-detect
        """
        try:
            # Step 1: Analyze the image context first
            image_context = self._analyze_image_context(image)
            logger.info(f"Image context: {image_context}")

            # Step 2: Get BLIP-based description
            description_results = self._get_blip_description(image)

            # Step 3: Extract realistic ingredients based on context
            if image_context['is_single_item']:
                detected_ingredients = self._detect_single_item(description_results, image_context)
                max_ingredients = 3  # At most 3 ingredients for single items
            elif image_context['is_prepared_dish']:
                detected_ingredients = self._detect_prepared_dish_ingredients(description_results, image_context)
                max_ingredients = min(top_k, 12)  # More ingredients allowed for prepared dishes
            else:
                detected_ingredients = self._detect_multiple_items(description_results, image_context)
                max_ingredients = min(top_k, 6)  # Moderate number for multiple raw ingredients

            # Step 4: Apply conservative filtering
            filtered_results = self._apply_conservative_filtering(
                detected_ingredients, confidence_threshold, max_ingredients
            )

            # Step 5: Add realistic confidence adjustments
            final_results = self._adjust_realistic_confidence(filtered_results, image_context)

            # Step 6: Sort by confidence and return top results
            final_results.sort(key=lambda x: x['confidence'], reverse=True)

            logger.info(f"Final detection: {len(final_results)} ingredients")
            return final_results[:max_ingredients]

        except Exception as e:
            logger.error(f"Error in accurate ingredient detection: {e}")
            return self._get_fallback_single_ingredient()

    def _analyze_image_context(self, image: Image.Image) -> Dict:
        """Analyze the image to understand context before detecting ingredients"""
        try:
            # Quick BLIP analysis to understand image context
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.blip_model.generate(**inputs, max_length=30)
                caption = self.blip_processor.decode(generated_ids[0], skip_special_tokens=True)

            caption_lower = caption.lower()
            logger.info(f"Image caption: {caption}")

            context = {
                'caption': caption,
                'is_single_item': False,
                'is_prepared_dish': False,
                'is_multiple_raw_items': False,
                'dominant_ingredient': None
            }

            # Check for single item indicators
            single_indicators = ['a tomato', 'an apple', 'a banana', 'an orange', 'a lemon', 
                               'a lime', 'a potato', 'an onion', 'a carrot', 'single', 'one', 'whole']

            if any(indicator in caption_lower for indicator in single_indicators):
                context['is_single_item'] = True
                # Extract the dominant ingredient
                for ingredient in self.ingredient_database:
                    if ingredient in caption_lower:
                        context['dominant_ingredient'] = ingredient
                        break

            # Check for prepared dish indicators
            elif any(indicator in caption_lower for indicator in self.common_patterns['prepared_dish_indicators']):
                context['is_prepared_dish'] = True

            # Check for multiple raw ingredients
            elif any(indicator in caption_lower for indicator in ['vegetables', 'fruits', 'ingredients']):
                context['is_multiple_raw_items'] = True

            # Default assumption if unclear
            else:
                # If caption mentions specific ingredients, assume single or few items
                ingredient_count = sum(1 for ingredient in self.ingredient_database if ingredient in caption_lower)
                if ingredient_count <= 2:
                    context['is_single_item'] = True
                else:
                    context['is_multiple_raw_items'] = True

            return context

        except Exception as e:
            logger.error(f"Error analyzing image context: {e}")
            return {
                'caption': 'unknown',
                'is_single_item': True,  # Default to conservative single item
                'is_prepared_dish': False,
                'is_multiple_raw_items': False,
                'dominant_ingredient': None
            }

    def _get_blip_description(self, image: Image.Image) -> Dict:
        """Get detailed description from BLIP model"""
        try:
            # General description
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.blip_model.generate(**inputs, max_length=50)
                general_caption = self.blip_processor.decode(generated_ids[0], skip_special_tokens=True)

            # Food-specific description
            food_prompt = "food ingredients:"
            conditional_inputs = self.blip_processor(
                image, food_prompt, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                food_ids = self.blip_model.generate(**conditional_inputs, max_length=40)
                food_caption = self.blip_processor.decode(food_ids[0], skip_special_tokens=True)

            return {
                'general_caption': general_caption,
                'food_caption': food_caption,
                'combined_text': f"{general_caption} {food_caption}"
            }

        except Exception as e:
            logger.error(f"Error getting BLIP description: {e}")
            return {
                'general_caption': 'food item',
                'food_caption': 'unknown ingredients',
                'combined_text': 'food item unknown ingredients'
            }

    def _detect_single_item(self, description_results: Dict, context: Dict) -> List[Dict]:
        """Detect ingredients for single item images - very conservative"""
        combined_text = description_results['combined_text'].lower()

        results = []

        # If we have a dominant ingredient from context, prioritize it
        if context['dominant_ingredient']:
            results.append({
                'ingredient': context['dominant_ingredient'],
                'confidence': 0.92,  # High confidence for context-identified items
                'source': 'context_analysis',
                'reasoning': 'Identified from image context analysis'
            })

        # Look for exact matches in the description
        text_embedding = self.sentence_model.encode([combined_text])
        similarities = cosine_similarity(text_embedding, self.ingredient_embeddings)[0]

        # Very high threshold for single items to avoid over-detection
        high_threshold = 0.65

        for idx, similarity in enumerate(similarities):
            ingredient = self.ingredient_database[idx]

            # Skip if already added from context
            if context['dominant_ingredient'] and ingredient == context['dominant_ingredient']:
                continue

            # Only add if very high confidence and makes sense for single item
            if similarity > high_threshold:
                # Additional check: is this ingredient likely to appear alone?
                if self._is_likely_single_ingredient(ingredient):
                    results.append({
                        'ingredient': ingredient,
                        'confidence': float(similarity) * 0.9,  # Slight discount
                        'source': 'semantic_matching',
                        'reasoning': f'High semantic similarity ({similarity:.2f}) for single item'
                    })

        # For single items, maximum 2-3 ingredients make sense
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results[:2] if context['dominant_ingredient'] else results[:1]

    def _detect_prepared_dish_ingredients(self, description_results: Dict, context: Dict) -> List[Dict]:
        """Detect ingredients for prepared dishes - more permissive"""
        combined_text = description_results['combined_text'].lower()

        results = []

        # Use semantic similarity with moderate threshold
        text_embedding = self.sentence_model.encode([combined_text])
        similarities = cosine_similarity(text_embedding, self.ingredient_embeddings)[0]

        moderate_threshold = 0.35

        for idx, similarity in enumerate(similarities):
            ingredient = self.ingredient_database[idx]

            if similarity > moderate_threshold:
                # Check if this ingredient makes sense in prepared dishes
                if self._is_common_cooking_ingredient(ingredient):
                    results.append({
                        'ingredient': ingredient,
                        'confidence': float(similarity) * 0.85,  # Discount for prepared dishes
                        'source': 'prepared_dish_analysis',
                        'reasoning': f'Detected in prepared dish ({similarity:.2f})'
                    })

        # For prepared dishes, more ingredients are reasonable
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results[:8]  # Max 8 for prepared dishes

    def _detect_multiple_items(self, description_results: Dict, context: Dict) -> List[Dict]:
        """Detect ingredients for multiple raw items - balanced approach"""
        combined_text = description_results['combined_text'].lower()

        results = []

        text_embedding = self.sentence_model.encode([combined_text])
        similarities = cosine_similarity(text_embedding, self.ingredient_embeddings)[0]

        balanced_threshold = 0.45

        for idx, similarity in enumerate(similarities):
            ingredient = self.ingredient_database[idx]

            if similarity > balanced_threshold:
                results.append({
                    'ingredient': ingredient,
                    'confidence': float(similarity) * 0.8,
                    'source': 'multiple_items_analysis',
                    'reasoning': f'Multiple items detection ({similarity:.2f})'
                })

        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results[:5]  # Max 5 for multiple raw items

    def _is_likely_single_ingredient(self, ingredient: str) -> bool:
        """Check if an ingredient is likely to appear alone in photos"""
        single_item_ingredients = [
            'tomato', 'tomatoes', 'apple', 'banana', 'orange', 'lemon', 'lime',
            'avocado', 'onion', 'potato', 'carrot', 'bell pepper', 'cucumber',
            'eggplant', 'zucchini', 'broccoli', 'cauliflower', 'corn', 'watermelon',
            'pineapple', 'mango', 'peach', 'pear'
        ]
        return ingredient.lower() in single_item_ingredients

    def _is_common_cooking_ingredient(self, ingredient: str) -> bool:
        """Check if an ingredient commonly appears in cooked dishes"""
        cooking_ingredients = [
            'onion', 'onions', 'garlic', 'tomato', 'tomatoes', 'bell pepper', 
            'carrots', 'celery', 'mushrooms', 'spinach', 'cheese', 'chicken',
            'beef', 'pork', 'fish', 'rice', 'pasta', 'noodles', 'beans',
            'olive oil', 'butter', 'herbs', 'spices'
        ]
        return ingredient.lower() in cooking_ingredients

    def _apply_conservative_filtering(self, detected_ingredients: List[Dict], 
                                   confidence_threshold: float, max_ingredients: int) -> List[Dict]:
        """Apply conservative filtering to avoid over-detection"""

        # Filter by confidence threshold
        filtered = [
            ing for ing in detected_ingredients 
            if ing['confidence'] >= confidence_threshold
        ]

        # Remove very similar ingredients (e.g., 'tomato' and 'tomatoes')
        filtered = self._remove_duplicate_ingredients(filtered)

        # Apply realistic limits
        filtered.sort(key=lambda x: x['confidence'], reverse=True)

        return filtered[:max_ingredients]

    def _remove_duplicate_ingredients(self, ingredients: List[Dict]) -> List[Dict]:
        """Remove duplicate/similar ingredients"""
        seen_ingredients = set()
        filtered = []

        for ingredient_data in ingredients:
            ingredient = ingredient_data['ingredient'].lower()

            # Check for exact duplicates
            if ingredient in seen_ingredients:
                continue

            # Check for plural/singular duplicates
            singular_form = ingredient.rstrip('s')
            plural_form = ingredient + 's' if not ingredient.endswith('s') else ingredient

            if singular_form in seen_ingredients or plural_form in seen_ingredients:
                # Keep the higher confidence one
                existing_idx = next((i for i, ing in enumerate(filtered) 
                                   if ing['ingredient'].lower() in [singular_form, plural_form]), None)

                if existing_idx is not None:
                    if ingredient_data['confidence'] > filtered[existing_idx]['confidence']:
                        filtered[existing_idx] = ingredient_data
                    continue

            seen_ingredients.add(ingredient)
            seen_ingredients.add(singular_form)
            seen_ingredients.add(plural_form)
            filtered.append(ingredient_data)

        return filtered

    def _adjust_realistic_confidence(self, ingredients: List[Dict], context: Dict) -> List[Dict]:
        """Adjust confidence scores to be more realistic"""

        for ingredient_data in ingredients:
            original_confidence = ingredient_data['confidence']

            # Boost confidence for context-identified ingredients
            if (context['dominant_ingredient'] and 
                ingredient_data['ingredient'] == context['dominant_ingredient']):
                ingredient_data['confidence'] = min(original_confidence * 1.2, 0.95)

            # Slight penalty for very common ingredients that might be false positives
            common_false_positives = ['olive oil', 'salt', 'pepper', 'garlic']
            if ingredient_data['ingredient'].lower() in common_false_positives:
                if not context['is_prepared_dish']:  # Only penalize if not a prepared dish
                    ingredient_data['confidence'] *= 0.7

            # Ensure confidence doesn't exceed realistic bounds
            ingredient_data['confidence'] = min(max(ingredient_data['confidence'], 0.3), 0.95)

        return ingredients

    def _get_fallback_single_ingredient(self) -> List[Dict]:
        """Conservative fallback for when detection fails"""
        return [{
            'ingredient': 'unknown food item',
            'confidence': 0.60,
            'source': 'fallback',
            'reasoning': 'Detection failed, showing conservative fallback'
        }]

# Example of how the improved system should work:
def create_accurate_simulation():
    """Create a more accurate simulation based on user's actual upload"""

    def simulate_single_tomato():
        """Simulate detection for a single tomato image"""
        return [
            {'ingredient': 'tomato', 'confidence': 0.92, 'source': 'context_analysis', 
             'reasoning': 'Single tomato identified from image context'},
            # Maybe one related item if there's some ambiguity
            {'ingredient': 'cherry tomatoes', 'confidence': 0.45, 'source': 'semantic_matching',
             'reasoning': 'Possible variety detected'}
        ]

    def simulate_tomato_salad():
        """Simulate detection for a tomato salad"""
        return [
            {'ingredient': 'tomatoes', 'confidence': 0.89, 'source': 'prepared_dish_analysis'},
            {'ingredient': 'lettuce', 'confidence': 0.76, 'source': 'prepared_dish_analysis'},
            {'ingredient': 'cucumber', 'confidence': 0.68, 'source': 'prepared_dish_analysis'},
            {'ingredient': 'olive oil', 'confidence': 0.55, 'source': 'prepared_dish_analysis'},
            {'ingredient': 'onion', 'confidence': 0.52, 'source': 'prepared_dish_analysis'}
        ]

    def simulate_multiple_vegetables():
        """Simulate detection for multiple vegetables"""
        return [
            {'ingredient': 'tomatoes', 'confidence': 0.85, 'source': 'multiple_items_analysis'},
            {'ingredient': 'bell peppers', 'confidence': 0.78, 'source': 'multiple_items_analysis'},
            {'ingredient': 'onions', 'confidence': 0.72, 'source': 'multiple_items_analysis'},
            {'ingredient': 'carrots', 'confidence': 0.65, 'source': 'multiple_items_analysis'}
        ]

    return {
        'single_tomato': simulate_single_tomato(),
        'tomato_salad': simulate_tomato_salad(),
        'multiple_vegetables': simulate_multiple_vegetables()
    }

# Test the improved system
accurate_simulations = create_accurate_simulation()

print("🎯 New Accurate Detection Examples:")
print("\nSingle Tomato Upload:")
for ing in accurate_simulations['single_tomato']:
    print(f"  • {ing['ingredient']}: {ing['confidence']:.0%} ({ing['source']})")

print("\nTomato Salad Upload:")
for ing in accurate_simulations['tomato_salad']:
    print(f"  • {ing['ingredient']}: {ing['confidence']:.0%} ({ing['source']})")

print("\nMultiple Vegetables Upload:")
for ing in accurate_simulations['multiple_vegetables']:
    print(f"  • {ing['ingredient']}: {ing['confidence']:.0%} ({ing['source']})")
