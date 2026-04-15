# Create example usage and API documentation
example_usage = '''#!/usr/bin/env python3
"""
Example usage of AI-Powered Multi-Sensory Recipe Generator

This file demonstrates how to use the different components programmatically
without the Streamlit interface.
"""

import numpy as np
from PIL import Image
import json
from datetime import datetime

# Import our custom modules
from ai_models import IngredientDetectionModel, RecipeGenerationModel
from rl_system import PersonalizationAgent, MultiArmedBanditRecommender

def example_ingredient_detection():
    """Example: Detect ingredients from an image"""
    print("🔍 Ingredient Detection Example")
    print("=" * 40)
    
    # Initialize the model
    detector = IngredientDetectionModel()
    
    # Create a sample image (in practice, load from file)
    # image = Image.open("path/to/your/food/image.jpg")
    
    # For demo, create a dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')
    
    # Detect ingredients
    detected_ingredients = detector.detect_ingredients(dummy_image, top_k=5)
    
    print("Detected ingredients:")
    for ingredient in detected_ingredients:
        print(f"  • {ingredient['ingredient']}: {ingredient['confidence']:.2%} confidence")
    
    return detected_ingredients

def example_recipe_generation():
    """Example: Generate a recipe from ingredients and preferences"""
    print("\\n📋 Recipe Generation Example")
    print("=" * 40)
    
    # Initialize the model
    generator = RecipeGenerationModel()
    
    # Sample ingredients (from detection or manual input)
    ingredients = ["chicken breast", "bell peppers", "onions", "garlic", "soy sauce"]
    
    # Sample preferences
    preferences = {
        'flavor': 'spicy',
        'textures': ['tender', 'crispy'],
        'dietary': 'None',
        'cooking_time': 30,
        'serving_size': 4
    }
    
    print(f"Ingredients: {', '.join(ingredients)}")
    print(f"Preferences: {preferences['flavor']} flavor, {preferences['cooking_time']} min max")
    
    # Generate recipe
    recipe = generator.generate_recipe(ingredients, preferences)
    
    print(f"\\nGenerated Recipe: {recipe['title']}")
    print(f"Description: {recipe['description']}")
    print(f"Cook time: {recipe['cook_time']} minutes")
    print(f"Servings: {recipe['servings']}")
    
    print("\\nIngredients:")
    for ing in recipe['ingredients'][:3]:  # Show first 3
        print(f"  • {ing['amount']} {ing['item']}")
    
    print(f"\\nSteps (showing first 3 of {len(recipe['steps'])}):")
    for step in recipe['steps'][:3]:
        timing = f" ({step['timing']} min)" if step['timing'] > 0 else ""
        print(f"  {step['step']}. {step['instruction']}{timing}")
    
    return recipe

def example_reinforcement_learning():
    """Example: Use RL agent for personalized recommendations"""
    print("\\n🤖 Reinforcement Learning Example")
    print("=" * 40)
    
    # Sample recipe database
    sample_recipes = [
        {
            "id": 1,
            "title": "Spicy Chicken Stir-Fry",
            "description": "A flavorful stir-fry with tender chicken and crisp vegetables",
            "cook_time": 20,
            "servings": 4,
            "difficulty": "Easy",
            "ingredients": [{"item": "chicken", "amount": "1 lb"}],
            "steps": [{"step": 1, "instruction": "Cook chicken", "timing": 10}]
        },
        {
            "id": 2,
            "title": "Creamy Mushroom Risotto",
            "description": "Rich and creamy risotto with mushrooms",
            "cook_time": 35,
            "servings": 4,
            "difficulty": "Medium", 
            "ingredients": [{"item": "rice", "amount": "1.5 cups"}],
            "steps": [{"step": 1, "instruction": "Cook risotto", "timing": 25}]
        }
    ]
    
    # Initialize RL agent
    agent = PersonalizationAgent(sample_recipes, model_type="PPO")
    
    print("Training RL agent...")
    # Train the agent (reduced timesteps for demo)
    agent.train(total_timesteps=1000)
    
    # User preferences
    user_preferences = {
        'flavor': 'spicy',
        'textures': ['tender', 'crispy'],
        'cooking_time': 30,
        'serving_size': 4
    }
    
    print(f"User preferences: {user_preferences}")
    
    # Get recommendations
    recommendations = agent.recommend_recipe(user_preferences, top_k=2)
    
    print("\\nPersonalized Recommendations:")
    for i, recipe in enumerate(recommendations, 1):
        score = recipe.get('recommendation_score', 0.95)
        print(f"  {i}. {recipe['title']} (Match: {score:.0%})")
    
    # Simulate user feedback
    print("\\nSimulating user feedback...")
    for recipe in recommendations:
        # Simulate feedback (in practice, this comes from user)
        feedback = {
            'satisfaction': np.random.uniform(0.6, 0.9),  # Positive feedback
            'taste': np.random.randint(3, 6),
            'difficulty': np.random.randint(2, 5)
        }
        
        recipe_idx = recipe.get('recipe_index', 0)
        agent.update_with_feedback(recipe_idx, feedback)
        
        print(f"    Feedback for '{recipe['title']}': {feedback['satisfaction']:.2f} satisfaction")
    
    # Get learning statistics
    stats = agent.get_learning_stats()
    print(f"\\nLearning Stats:")
    print(f"  • Total interactions: {stats['total_interactions']}")
    print(f"  • Average satisfaction: {stats['average_satisfaction']:.2f}")
    print(f"  • Recipe diversity: {stats['recipe_diversity']:.2%}")
    
    return agent

def example_multi_armed_bandit():
    """Example: Use Multi-Armed Bandit for recipe recommendations"""
    print("\\n🎰 Multi-Armed Bandit Example")
    print("=" * 40)
    
    # Sample recipes
    recipes = [
        {"title": "Pasta Carbonara", "difficulty": "Medium"},
        {"title": "Chicken Curry", "difficulty": "Easy"},
        {"title": "Beef Wellington", "difficulty": "Hard"}
    ]
    
    # Initialize bandit
    bandit = MultiArmedBanditRecommender(recipes)
    
    print(f"Testing {len(recipes)} recipes with bandit algorithm...")
    
    # Simulate user interactions
    n_rounds = 100
    total_reward = 0
    
    for round_num in range(n_rounds):
        # Get user context (simplified)
        user_context = np.random.random(11)  # Random context vector
        
        # Get recommendation
        recommended_indices = bandit.recommend(user_context, n_recommendations=1)
        recipe_idx = recommended_indices[0]
        
        # Simulate user feedback (reward)
        # Harder recipes get lower rewards on average
        if recipes[recipe_idx]["difficulty"] == "Easy":
            reward = np.random.beta(8, 3)  # High success rate
        elif recipes[recipe_idx]["difficulty"] == "Medium":
            reward = np.random.beta(6, 4)  # Medium success rate
        else:  # Hard
            reward = np.random.beta(3, 7)  # Low success rate
        
        # Update bandit
        bandit.update(recipe_idx, reward)
        total_reward += reward
        
        if (round_num + 1) % 25 == 0:
            avg_reward = total_reward / (round_num + 1)
            print(f"    Round {round_num + 1}: Average reward = {avg_reward:.3f}")
    
    # Get final statistics
    stats = bandit.get_recipe_stats()
    
    print(f"\\nFinal Bandit Statistics:")
    print(f"  • Best recipe: {recipes[stats['best_recipe_idx']]['title']}")
    print(f"  • Best success rate: {stats['best_success_rate']:.2%}")
    print(f"  • Average success rate: {stats['average_success_rate']:.2%}")
    print(f"  • Total trials: {stats['total_trials']}")
    
    print("\\nRecipe rankings (best to worst):")
    for i, recipe_idx in enumerate(stats['recipe_rankings'], 1):
        recipe_title = recipes[recipe_idx]['title']
        print(f"    {i}. {recipe_title}")
    
    return bandit

def example_full_pipeline():
    """Example: Complete pipeline from image to cooking"""
    print("\\n🔄 Full Pipeline Example")
    print("=" * 50)
    
    # Step 1: Ingredient Detection
    print("Step 1: Detecting ingredients from image...")
    detected_ingredients = example_ingredient_detection()
    ingredient_names = [ing['ingredient'] for ing in detected_ingredients]
    
    # Step 2: Set user preferences
    print("\\nStep 2: Setting user preferences...")
    preferences = {
        'flavor': 'spicy',
        'textures': ['crispy', 'tender'],
        'dietary': 'None',
        'cooking_time': 25,
        'serving_size': 4
    }
    print(f"Preferences: {preferences}")
    
    # Step 3: Generate recipe
    print("\\nStep 3: Generating personalized recipe...")
    generator = RecipeGenerationModel()
    recipe = generator.generate_recipe(ingredient_names, preferences)
    
    # Step 4: Simulate cooking with timers
    print(f"\\nStep 4: Starting cooking '{recipe['title']}'...")
    
    total_time = 0
    for step in recipe['steps']:
        step_time = step['timing']
        print(f"  Step {step['step']}: {step['instruction']}")
        
        if step_time > 0:
            print(f"    ⏱️  Timer: {step_time} minutes")
            total_time += step_time
    
    print(f"\\nTotal estimated cooking time: {total_time} minutes")
    
    # Step 5: Simulate user feedback
    print("\\nStep 5: Collecting user feedback...")
    feedback = {
        'overall_satisfaction': np.random.randint(4, 6),  # 4-5 stars
        'taste': np.random.randint(3, 6),
        'difficulty': np.random.randint(2, 5),
        'time_accuracy': np.random.randint(3, 6),
        'would_cook_again': np.random.choice(['Yes', 'Maybe']),
        'timestamp': datetime.now()
    }
    
    print(f"User feedback: {feedback['overall_satisfaction']}/5 stars overall")
    print(f"Taste: {feedback['taste']}/5, Would cook again: {feedback['would_cook_again']}")
    
    print("\\n✅ Complete pipeline executed successfully!")
    return {
        'ingredients': detected_ingredients,
        'recipe': recipe,
        'feedback': feedback
    }

def save_example_data():
    """Save example data to files for testing"""
    print("\\n💾 Saving Example Data")
    print("=" * 30)
    
    # Sample data for testing
    sample_data = {
        'ingredients': [
            {'ingredient': 'tomatoes', 'confidence': 0.89},
            {'ingredient': 'chicken breast', 'confidence': 0.85},
            {'ingredient': 'onions', 'confidence': 0.78},
            {'ingredient': 'garlic', 'confidence': 0.82}
        ],
        'preferences': {
            'flavor': 'spicy',
            'textures': ['crispy', 'tender'],
            'dietary': 'None',
            'cooking_time': 30,
            'serving_size': 4
        },
        'sample_recipes': [
            {
                "id": 1,
                "title": "Spicy Tomato Chicken",
                "description": "Tender chicken in spicy tomato sauce",
                "cook_time": 25,
                "servings": 4,
                "difficulty": "Easy",
                "ingredients": [
                    {"item": "Chicken breast", "amount": "1 lb", "prep": "diced"},
                    {"item": "Tomatoes", "amount": "2 large", "prep": "chopped"},
                    {"item": "Onions", "amount": "1 medium", "prep": "sliced"},
                    {"item": "Garlic", "amount": "3 cloves", "prep": "minced"}
                ],
                "steps": [
                    {"step": 1, "instruction": "Heat oil in pan", "timing": 2, "type": "prep"},
                    {"step": 2, "instruction": "Cook chicken until done", "timing": 8, "type": "cooking"},
                    {"step": 3, "instruction": "Add vegetables and cook", "timing": 10, "type": "cooking"},
                    {"step": 4, "instruction": "Add spices and simmer", "timing": 5, "type": "cooking"}
                ],
                "nutrition": {"calories": 320, "protein": "28g", "carbs": "12g", "fat": "18g"}
            }
        ]
    }
    
    # Save to JSON file
    with open("example_data.json", "w") as f:
        json.dump(sample_data, f, indent=2, default=str)
    
    print("Example data saved to 'example_data.json'")
    
    # Save configuration file
    config = {
        "model_settings": {
            "ingredient_detection_model": "Salesforce/blip-image-captioning-base",
            "sentence_model": "all-MiniLM-L6-v2",
            "detection_threshold": 0.3,
            "max_ingredients": 10
        },
        "rl_settings": {
            "algorithm": "PPO",
            "learning_rate": 3e-4,
            "training_timesteps": 10000,
            "reward_weights": {
                "satisfaction": 1.0,
                "diversity": 0.3,
                "novelty": 0.2
            }
        },
        "app_settings": {
            "default_cooking_time": 30,
            "default_serving_size": 4,
            "timer_refresh_rate": 1.0,
            "max_feedback_history": 100
        }
    }
    
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Configuration saved to 'config.json'")

if __name__ == "__main__":
    print("🍳 AI Recipe Generator - Example Usage")
    print("=" * 60)
    
    try:
        # Run examples
        save_example_data()
        
        # Note: These examples require the models to be loaded
        # Uncomment to run (requires setup.py to be run first)
        
        # example_ingredient_detection()
        # example_recipe_generation()
        # example_reinforcement_learning()
        # example_multi_armed_bandit()
        # example_full_pipeline()
        
        print("\\n🎉 All examples completed successfully!")
        print("\\nTo run the full examples, uncomment the function calls above")
        print("and ensure you've run 'python setup.py' first to install requirements.")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure to run 'python setup.py' first to install requirements.")
'''

with open("example_usage.py", "w") as f:
    f.write(example_usage)

print("Example usage file created successfully!")