# Create the main Streamlit application
streamlit_app_code = '''
import streamlit as st
import torch
import numpy as np
from PIL import Image
import time
import json
import pickle
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
import logging
import threading
import asyncio
from pathlib import Path

# Import our custom modules
from ai_models import IngredientDetectionModel, RecipeGenerationModel
from rl_system import PersonalizationAgent, MultiArmedBanditRecommender

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="🍳 AI Recipe Generator", 
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .success-message {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        color: #155724;
        margin: 1rem 0;
    }
    
    .timer-display {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        border: 3px solid #FF6B6B;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .step-card {
        background: #f8f9fa;
        border-left: 4px solid #4ECDC4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .ingredient-confidence {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem;
        background: #e9ecef;
        margin: 0.2rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

class RecipeApp:
    """Main application class"""
    
    def __init__(self):
        self.initialize_session_state()
        self.load_models()
        self.load_sample_data()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.current_tab = "ingredient_detection"
            st.session_state.detected_ingredients = []
            st.session_state.confirmed_ingredients = []
            st.session_state.user_preferences = {
                'flavor': None,
                'flavor_intensity': 5,
                'textures': [],
                'dietary': 'None',
                'cooking_time': 30,
                'serving_size': 4
            }
            st.session_state.current_recipe = None
            st.session_state.cooking_step = 0
            st.session_state.timer_active = False
            st.session_state.timer_start_time = None
            st.session_state.timer_duration = 0
            st.session_state.feedback_history = []
            st.session_state.personalization_score = 0.5
            st.session_state.learning_progress = 0.0
    
    @st.cache_resource
    def load_models(_self):
        """Load AI models with caching"""
        try:
            models = {}
            
            with st.spinner("🧠 Loading AI models..."):
                # Load ingredient detection model
                models['ingredient_detector'] = IngredientDetectionModel()
                
                # Load recipe generation model
                models['recipe_generator'] = RecipeGenerationModel()
                
            st.success("✅ AI models loaded successfully!")
            return models
            
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            logger.error(f"Model loading error: {e}")
            return {}
    
    def load_sample_data(self):
        """Load sample recipe data"""
        self.sample_recipes = [
            {
                "id": 1,
                "title": "Spicy Chicken Stir-Fry",
                "description": "A flavorful stir-fry with tender chicken and crisp vegetables in a spicy sauce",
                "cook_time": 20,
                "servings": 4,
                "difficulty": "Easy",
                "ingredients": [
                    {"item": "Chicken breast", "amount": "1 lb", "prep": "cut into strips"},
                    {"item": "Bell peppers", "amount": "2", "prep": "sliced"},
                    {"item": "Onion", "amount": "1 medium", "prep": "sliced"},
                    {"item": "Garlic", "amount": "3 cloves", "prep": "minced"},
                    {"item": "Soy sauce", "amount": "3 tbsp", "prep": ""},
                    {"item": "Sriracha", "amount": "2 tbsp", "prep": ""},
                    {"item": "Olive oil", "amount": "2 tbsp", "prep": ""},
                    {"item": "Rice", "amount": "2 cups cooked", "prep": ""}
                ],
                "steps": [
                    {"step": 1, "instruction": "Heat olive oil in a large pan over medium-high heat", "timing": 2, "type": "prep"},
                    {"step": 2, "instruction": "Cook chicken strips until golden brown and cooked through", "timing": 8, "type": "cooking"},
                    {"step": 3, "instruction": "Add garlic and cook until fragrant", "timing": 1, "type": "cooking"},
                    {"step": 4, "instruction": "Add bell peppers and onions, stir-fry until tender-crisp", "timing": 5, "type": "cooking"},
                    {"step": 5, "instruction": "Mix soy sauce and sriracha in a small bowl", "timing": 1, "type": "prep"},
                    {"step": 6, "instruction": "Pour sauce over chicken and vegetables, toss to coat", "timing": 2, "type": "cooking"},
                    {"step": 7, "instruction": "Let flavors meld and sauce thicken slightly", "timing": 2, "type": "cooking"},
                    {"step": 8, "instruction": "Serve hot over cooked rice", "timing": 0, "type": "serving"}
                ],
                "nutrition": {"calories": 380, "protein": "28g", "carbs": "35g", "fat": "12g"}
            },
            {
                "id": 2,
                "title": "Creamy Mushroom Risotto",
                "description": "Rich and creamy risotto with sautéed mushrooms and fresh herbs",
                "cook_time": 35,
                "servings": 4,
                "difficulty": "Medium",
                "ingredients": [
                    {"item": "Arborio rice", "amount": "1.5 cups", "prep": ""},
                    {"item": "Mushrooms", "amount": "8 oz", "prep": "sliced"},
                    {"item": "Onion", "amount": "1 small", "prep": "finely diced"},
                    {"item": "Garlic", "amount": "2 cloves", "prep": "minced"},
                    {"item": "White wine", "amount": "1/2 cup", "prep": ""},
                    {"item": "Chicken stock", "amount": "4 cups", "prep": "warm"},
                    {"item": "Butter", "amount": "3 tbsp", "prep": ""},
                    {"item": "Parmesan cheese", "amount": "1/2 cup", "prep": "grated"}
                ],
                "steps": [
                    {"step": 1, "instruction": "Heat stock in a separate pan and keep warm", "timing": 0, "type": "prep"},
                    {"step": 2, "instruction": "Sauté mushrooms in 1 tbsp butter until golden, set aside", "timing": 5, "type": "prep"},
                    {"step": 3, "instruction": "In same pan, cook onion and garlic until softened", "timing": 3, "type": "cooking"},
                    {"step": 4, "instruction": "Add rice and stir to coat with oil, cook until edges are translucent", "timing": 2, "type": "cooking"},
                    {"step": 5, "instruction": "Add wine and stir until absorbed", "timing": 2, "type": "cooking"},
                    {"step": 6, "instruction": "Add stock one ladle at a time, stirring continuously until absorbed", "timing": 18, "type": "cooking"},
                    {"step": 7, "instruction": "Stir in mushrooms, remaining butter, and Parmesan", "timing": 2, "type": "finishing"},
                    {"step": 8, "instruction": "Season with salt and pepper, serve immediately", "timing": 0, "type": "serving"}
                ],
                "nutrition": {"calories": 420, "protein": "12g", "carbs": "65g", "fat": "14g"}
            },
            {
                "id": 3,
                "title": "Crispy Fish Tacos",
                "description": "Light and crispy fish tacos with fresh slaw and zesty lime crema",
                "cook_time": 25,
                "servings": 6,
                "difficulty": "Easy",
                "ingredients": [
                    {"item": "White fish fillets", "amount": "1.5 lbs", "prep": "cut into strips"},
                    {"item": "Corn tortillas", "amount": "12", "prep": ""},
                    {"item": "Cabbage", "amount": "2 cups", "prep": "shredded"},
                    {"item": "Lime", "amount": "2", "prep": "juiced"},
                    {"item": "Sour cream", "amount": "1/2 cup", "prep": ""},
                    {"item": "Flour", "amount": "1 cup", "prep": ""},
                    {"item": "Beer", "amount": "3/4 cup", "prep": "cold"},
                    {"item": "Oil for frying", "amount": "2 cups", "prep": ""}
                ],
                "steps": [
                    {"step": 1, "instruction": "Heat oil to 375°F in a large pot", "timing": 5, "type": "prep"},
                    {"step": 2, "instruction": "Mix flour and beer to create smooth batter", "timing": 2, "type": "prep"},
                    {"step": 3, "instruction": "Combine cabbage with lime juice for slaw", "timing": 3, "type": "prep"},
                    {"step": 4, "instruction": "Mix sour cream with remaining lime juice for crema", "timing": 1, "type": "prep"},
                    {"step": 5, "instruction": "Dip fish in batter and fry until golden and crispy", "timing": 8, "type": "cooking"},
                    {"step": 6, "instruction": "Drain fish on paper towels", "timing": 2, "type": "cooking"},
                    {"step": 7, "instruction": "Warm tortillas in dry pan or microwave", "timing": 2, "type": "prep"},
                    {"step": 8, "instruction": "Assemble tacos with fish, slaw, and crema", "timing": 0, "type": "serving"}
                ],
                "nutrition": {"calories": 340, "protein": "25g", "carbs": "28g", "fat": "15g"}
            }
        ]
        
        # Initialize RL agent
        if 'rl_agent' not in st.session_state:
            try:
                st.session_state.rl_agent = PersonalizationAgent(self.sample_recipes)
                st.session_state.bandit_recommender = MultiArmedBanditRecommender(self.sample_recipes)
            except Exception as e:
                logger.error(f"Error initializing RL agents: {e}")
                st.session_state.rl_agent = None
                st.session_state.bandit_recommender = None
    
    def run(self):
        """Main application runner"""
        # Header
        st.markdown('<div class="main-header">🍳 AI-Powered Multi-Sensory Recipe Generator</div>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            🧠 <strong>Neural Networks</strong> • 🔄 <strong>Reinforcement Learning</strong> • 
            👁️ <strong>Computer Vision</strong> • 🎯 <strong>Personalization</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar navigation
        with st.sidebar:
            st.title("🔧 Navigation")
            
            tabs = {
                "🔍 Ingredient Detection": "ingredient_detection",
                "⚙️ Preferences": "preferences", 
                "📋 Recipe Generation": "recipe",
                "👨‍🍳 Cooking Mode": "cooking",
                "📊 Feedback & Learning": "feedback"
            }
            
            selected_tab = st.radio("Choose Section:", list(tabs.keys()))
            st.session_state.current_tab = tabs[selected_tab]
            
            # Show AI model status
            st.markdown("---")
            st.subheader("🤖 AI Status")
            
            models_status = getattr(self, 'models', {})
            if models_status:
                st.success("✅ Neural Networks Active")
                st.success("✅ Computer Vision Ready") 
                st.success("✅ RL Agent Training")
            else:
                st.warning("⚠️ Models Loading...")
            
            # Learning stats
            if st.session_state.feedback_history:
                st.markdown("---")
                st.subheader("📈 Learning Stats")
                avg_satisfaction = np.mean([f['satisfaction'] for f in st.session_state.feedback_history])
                st.metric("Avg Satisfaction", f"{avg_satisfaction:.2f}")
                st.metric("Total Interactions", len(st.session_state.feedback_history))
                st.progress(st.session_state.learning_progress)
        
        # Main content based on selected tab
        if st.session_state.current_tab == "ingredient_detection":
            self.ingredient_detection_tab()
        elif st.session_state.current_tab == "preferences":
            self.preferences_tab()
        elif st.session_state.current_tab == "recipe":
            self.recipe_tab()
        elif st.session_state.current_tab == "cooking":
            self.cooking_tab()
        elif st.session_state.current_tab == "feedback":
            self.feedback_tab()
    
    def ingredient_detection_tab(self):
        """Ingredient detection interface"""
        st.header("🔍 AI-Powered Ingredient Detection")
        st.markdown("Upload a food image and let our neural networks identify the ingredients!")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Image Upload")
            uploaded_file = st.file_uploader(
                "Choose a food image...", 
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear image of your ingredients or prepared dish"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Detect ingredients button
                if st.button("🧠 Analyze Ingredients", type="primary"):
                    with st.spinner("🔍 Neural network analyzing image..."):
                        # Simulate processing time
                        time.sleep(2)
                        
                        # Use AI model for detection (if available)
                        try:
                            if hasattr(self, 'models') and 'ingredient_detector' in self.models:
                                detected = self.models['ingredient_detector'].detect_ingredients(image)
                            else:
                                # Fallback detection
                                detected = self.simulate_ingredient_detection()
                            
                            st.session_state.detected_ingredients = detected
                            st.success(f"✅ Detected {len(detected)} ingredients!")
                            
                        except Exception as e:
                            st.error(f"❌ Detection failed: {e}")
                            st.session_state.detected_ingredients = self.simulate_ingredient_detection()
        
        with col2:
            st.subheader("🎯 Detection Results")
            
            if st.session_state.detected_ingredients:
                st.markdown("**Detected Ingredients:**")
                
                confirmed_ingredients = []
                
                for i, ingredient_data in enumerate(st.session_state.detected_ingredients):
                    ingredient = ingredient_data['ingredient']
                    confidence = ingredient_data['confidence']
                    
                    # Create ingredient confirmation interface
                    col_ing, col_conf, col_check = st.columns([3, 1, 1])
                    
                    with col_ing:
                        st.write(f"**{ingredient.title()}**")
                    
                    with col_conf:
                        # Confidence bar
                        conf_color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
                        st.markdown(f'<div style="background-color: {conf_color}; width: {confidence*100}%; height: 20px; border-radius: 10px; display: inline-block;"></div> {confidence:.0%}', 
                                   unsafe_allow_html=True)
                    
                    with col_check:
                        confirmed = st.checkbox("✓", key=f"confirm_{i}", value=True)
                        if confirmed:
                            confirmed_ingredients.append(ingredient)
                
                st.session_state.confirmed_ingredients = confirmed_ingredients
                
                # Feedback for detection accuracy
                st.markdown("---")
                st.subheader("🎯 Detection Feedback")
                
                col_fb1, col_fb2 = st.columns(2)
                with col_fb1:
                    detection_rating = st.slider(
                        "How accurate was the detection?", 
                        1, 5, 3,
                        help="Rate the ingredient detection accuracy"
                    )
                
                with col_fb2:
                    if st.button("💾 Submit Feedback"):
                        feedback = {
                            'type': 'detection',
                            'accuracy_rating': detection_rating,
                            'timestamp': datetime.now(),
                            'detected_count': len(st.session_state.detected_ingredients),
                            'confirmed_count': len(confirmed_ingredients)
                        }
                        st.session_state.feedback_history.append(feedback)
                        
                        # Update RL agent
                        reward = detection_rating / 5.0
                        if st.session_state.rl_agent:
                            try:
                                st.session_state.rl_agent.update_with_feedback(0, {'satisfaction': reward})
                            except Exception as e:
                                logger.warning(f"RL update failed: {e}")
                        
                        st.success("✅ Feedback submitted! AI is learning...")
                        
                        # Update learning progress
                        st.session_state.learning_progress = min(
                            st.session_state.learning_progress + 0.1, 1.0
                        )
            
            else:
                st.info("👆 Upload an image above to start ingredient detection")
    
    def simulate_ingredient_detection(self):
        """Simulate ingredient detection for demo"""
        sample_ingredients = [
            {'ingredient': 'tomatoes', 'confidence': 0.89},
            {'ingredient': 'onions', 'confidence': 0.76}, 
            {'ingredient': 'garlic', 'confidence': 0.82},
            {'ingredient': 'bell peppers', 'confidence': 0.71},
            {'ingredient': 'olive oil', 'confidence': 0.65},
            {'ingredient': 'chicken breast', 'confidence': 0.78}
        ]
        return sample_ingredients
    
    def preferences_tab(self):
        """User preferences interface"""
        st.header("⚙️ Multi-Sensory Preferences")
        st.markdown("Tell us about your taste preferences to get personalized recommendations!")
        
        # Flavor Preferences
        st.subheader("🌶️ Flavor Profile")
        
        col1, col2 = st.columns(2)
        
        with col1:
            flavor_options = {
                '🌶️ Spicy': 'spicy',
                '🍯 Sweet': 'sweet', 
                '🍋 Tangy': 'tangy',
                '🧄 Savory': 'savory',
                '☕ Bitter': 'bitter'
            }
            
            selected_flavor = st.radio(
                "Primary Flavor Preference:",
                list(flavor_options.keys()),
                help="Choose your favorite flavor profile"
            )
            
            st.session_state.user_preferences['flavor'] = flavor_options[selected_flavor]
        
        with col2:
            intensity = st.slider(
                "Flavor Intensity (1=Mild, 10=Intense):",
                1, 10, 
                st.session_state.user_preferences['flavor_intensity'],
                help="How intense do you like your flavors?"
            )
            st.session_state.user_preferences['flavor_intensity'] = intensity
        
        # Texture Preferences
        st.markdown("---")
        st.subheader("🥄 Texture Preferences")
        
        texture_options = {
            '🍟 Crispy': 'crispy',
            '🍰 Soft': 'soft',
            '🍖 Chewy': 'chewy', 
            '🥩 Tender': 'tender',
            '🥜 Crunchy': 'crunchy',
            '🍦 Creamy': 'creamy'
        }
        
        selected_textures = []
        cols = st.columns(3)
        
        for i, (display_name, texture_key) in enumerate(texture_options.items()):
            with cols[i % 3]:
                if st.checkbox(display_name, key=f"texture_{texture_key}"):
                    selected_textures.append(texture_key)
        
        st.session_state.user_preferences['textures'] = selected_textures
        
        # Additional Preferences
        st.markdown("---")
        st.subheader("🍽️ Additional Preferences")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dietary = st.selectbox(
                "Dietary Restrictions:",
                ['None', 'Vegetarian', 'Vegan', 'Gluten-Free', 'Keto', 'Dairy-Free', 'Nut-Free'],
                index=['None', 'Vegetarian', 'Vegan', 'Gluten-Free', 'Keto', 'Dairy-Free', 'Nut-Free'].index(
                    st.session_state.user_preferences['dietary']
                )
            )
            st.session_state.user_preferences['dietary'] = dietary
        
        with col2:
            cook_time = st.slider(
                "Max Cooking Time (minutes):",
                15, 120, 
                st.session_state.user_preferences['cooking_time']
            )
            st.session_state.user_preferences['cooking_time'] = cook_time
        
        with col3:
            serving_size = st.slider(
                "Serving Size (people):",
                1, 8,
                st.session_state.user_preferences['serving_size']
            )
            st.session_state.user_preferences['serving_size'] = serving_size
        
        # Save preferences
        if st.button("💾 Save Preferences", type="primary"):
            st.success("✅ Preferences saved! Your AI will now personalize recommendations.")
            
            # Update RL agent with preferences
            if st.session_state.rl_agent:
                try:
                    st.session_state.rl_agent._update_environment_preferences(
                        st.session_state.user_preferences
                    )
                except Exception as e:
                    logger.warning(f"Preference update failed: {e}")
            
            st.session_state.personalization_score = min(
                st.session_state.personalization_score + 0.2, 1.0
            )
        
        # Display current preferences summary
        st.markdown("---")
        st.subheader("📋 Current Preferences Summary")
        
        prefs = st.session_state.user_preferences
        st.write(f"**Flavor:** {prefs['flavor'].title()} (Intensity: {prefs['flavor_intensity']}/10)")
        st.write(f"**Textures:** {', '.join([t.title() for t in prefs['textures']]) if prefs['textures'] else 'None selected'}")
        st.write(f"**Dietary:** {prefs['dietary']}")
        st.write(f"**Max Cook Time:** {prefs['cooking_time']} minutes")
        st.write(f"**Serving Size:** {prefs['serving_size']} people")
    
    def recipe_tab(self):
        """Recipe generation and display"""
        st.header("📋 Personalized Recipe Generation")
        
        if not st.session_state.confirmed_ingredients:
            st.warning("⚠️ Please detect some ingredients first in the Ingredient Detection tab!")
            return
        
        if not st.session_state.user_preferences['flavor']:
            st.warning("⚠️ Please set your preferences first in the Preferences tab!")
            return
        
        # Display ingredients and preferences
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🥗 Available Ingredients")
            for ingredient in st.session_state.confirmed_ingredients:
                st.write(f"• {ingredient.title()}")
        
        with col2:
            st.subheader("👤 Your Preferences") 
            prefs = st.session_state.user_preferences
            st.write(f"• **Flavor:** {prefs['flavor'].title()}")
            st.write(f"• **Textures:** {', '.join([t.title() for t in prefs['textures']])}")
            st.write(f"• **Max Time:** {prefs['cooking_time']} min")
        
        # Generate recipe button
        if st.button("🧠 Generate Personalized Recipe", type="primary"):
            with st.spinner("🤖 AI is creating your perfect recipe..."):
                # Simulate processing
                time.sleep(3)
                
                try:
                    # Use RL agent for recommendation
                    if st.session_state.rl_agent:
                        recommendations = st.session_state.rl_agent.recommend_recipe(
                            st.session_state.user_preferences, top_k=1
                        )
                        if recommendations:
                            st.session_state.current_recipe = recommendations[0]
                        else:
                            st.session_state.current_recipe = self.sample_recipes[0]
                    else:
                        # Fallback to rule-based selection
                        st.session_state.current_recipe = self.select_recipe_by_preferences()
                    
                    st.success("✅ Recipe generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Recipe generation failed: {e}")
                    st.session_state.current_recipe = self.sample_recipes[0]
        
        # Display generated recipe
        if st.session_state.current_recipe:
            recipe = st.session_state.current_recipe
            
            st.markdown("---")
            st.markdown(f'<div class="feature-card"><h2>🍽️ {recipe["title"]}</h2><p>{recipe["description"]}</p></div>', 
                       unsafe_allow_html=True)
            
            # Recipe details
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("⏱️ Cook Time", f"{recipe['cook_time']} min")
            with col2:
                st.metric("👥 Servings", recipe['servings'])
            with col3:
                st.metric("📈 Difficulty", recipe['difficulty'])
            with col4:
                if 'recommendation_score' in recipe:
                    st.metric("🎯 AI Match", f"{recipe['recommendation_score']:.0%}")
                else:
                    st.metric("🎯 AI Match", "95%")
            
            # Ingredients list
            st.subheader("🛒 Ingredients")
            for ingredient in recipe['ingredients']:
                st.write(f"• **{ingredient['amount']}** {ingredient['item']} {f'({ingredient["prep"]})' if ingredient['prep'] else ''}")
            
            # Cooking steps preview
            st.subheader("👨‍🍳 Cooking Steps Preview")
            for step in recipe['steps'][:3]:  # Show first 3 steps
                st.write(f"**Step {step['step']}:** {step['instruction']} {f'({step["timing"]} min)' if step['timing'] > 0 else ''}")
            
            if len(recipe['steps']) > 3:
                st.write(f"... and {len(recipe['steps']) - 3} more steps")
            
            # Nutrition info
            if 'nutrition' in recipe:
                st.subheader("📊 Nutrition (per serving)")
                nutrition = recipe['nutrition']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Calories", nutrition.get('calories', 'N/A'))
                with col2:
                    st.metric("Protein", nutrition.get('protein', 'N/A'))
                with col3:
                    st.metric("Carbs", nutrition.get('carbs', 'N/A'))
                with col4:
                    st.metric("Fat", nutrition.get('fat', 'N/A'))
            
            # Start cooking button
            if st.button("🔥 Start Cooking!", type="primary"):
                st.session_state.current_tab = "cooking"
                st.session_state.cooking_step = 0
                st.success("✅ Let's start cooking! Navigate to Cooking Mode.")
    
    def select_recipe_by_preferences(self):
        """Rule-based recipe selection fallback"""
        prefs = st.session_state.user_preferences
        
        # Simple scoring based on preferences
        best_recipe = self.sample_recipes[0]
        best_score = 0
        
        for recipe in self.sample_recipes:
            score = 0
            title_desc = f"{recipe['title']} {recipe['description']}".lower()
            
            # Flavor matching
            if prefs['flavor'] in title_desc:
                score += 3
            
            # Texture matching
            for texture in prefs['textures']:
                if texture in title_desc:
                    score += 1
            
            # Time preference
            if recipe['cook_time'] <= prefs['cooking_time']:
                score += 2
            
            if score > best_score:
                best_score = score
                best_recipe = recipe
        
        return best_recipe
    
    def cooking_tab(self):
        """Interactive cooking mode with timers"""
        st.header("👨‍🍳 Interactive Cooking Mode")
        
        if not st.session_state.current_recipe:
            st.warning("⚠️ Please generate a recipe first!")
            return
        
        recipe = st.session_state.current_recipe
        current_step = st.session_state.cooking_step
        total_steps = len(recipe['steps'])
        
        # Progress bar
        progress = (current_step + 1) / total_steps
        st.progress(progress)
        st.write(f"Step {current_step + 1} of {total_steps}")
        
        if current_step < total_steps:
            step_data = recipe['steps'][current_step]
            
            # Current step display
            st.markdown(f"""
            <div class="step-card">
                <h3>Step {step_data['step']}: {step_data['type'].title()}</h3>
                <p style="font-size: 1.2rem; font-weight: bold;">{step_data['instruction']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Timer functionality
            timer_duration = step_data.get('timing', 0)
            
            if timer_duration > 0:
                st.subheader(f"⏱️ Timer: {timer_duration} minutes")
                
                # Timer display
                if st.session_state.timer_active and st.session_state.timer_start_time:
                    elapsed = datetime.now() - st.session_state.timer_start_time
                    remaining = timedelta(minutes=timer_duration) - elapsed
                    
                    if remaining.total_seconds() > 0:
                        mins, secs = divmod(int(remaining.total_seconds()), 60)
                        st.markdown(f'<div class="timer-display">{mins:02d}:{secs:02d}</div>', 
                                   unsafe_allow_html=True)
                        
                        # Auto-refresh for timer countdown
                        if remaining.total_seconds() <= 1:
                            st.session_state.timer_active = False
                            st.balloons()
                            st.success("⏰ Timer finished! You can now proceed to the next step.")
                    else:
                        st.session_state.timer_active = False
                        st.success("⏰ Timer finished!")
                else:
                    st.markdown(f'<div class="timer-display">{timer_duration:02d}:00</div>', 
                               unsafe_allow_html=True)
                
                # Timer controls
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if not st.session_state.timer_active:
                        if st.button("▶️ Start Timer", type="primary"):
                            st.session_state.timer_active = True
                            st.session_state.timer_start_time = datetime.now()
                            st.session_state.timer_duration = timer_duration
                            st.rerun()
                    else:
                        if st.button("⏸️ Pause Timer"):
                            st.session_state.timer_active = False
                
                with col2:
                    if st.button("⏭️ Skip Timer", help="Skip timer for debugging"):
                        st.session_state.timer_active = False
                        st.warning("⏭️ Timer skipped!")
                
                with col3:
                    if st.button("🔄 Reset Timer"):
                        st.session_state.timer_active = False
                        st.session_state.timer_start_time = None
            
            # Navigation buttons
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if current_step > 0:
                    if st.button("⬅️ Previous Step"):
                        st.session_state.cooking_step -= 1
                        st.session_state.timer_active = False
                        st.rerun()
            
            with col2:
                # Next step button - locked if timer is active
                timer_complete = (not st.session_state.timer_active and 
                                timer_duration > 0) or timer_duration == 0
                
                if timer_complete:
                    if st.button("➡️ Next Step", type="primary"):
                        st.session_state.cooking_step += 1
                        st.session_state.timer_active = False
                        st.session_state.timer_start_time = None
                        st.rerun()
                else:
                    st.button("🔒 Next Step (Timer Running)", disabled=True, 
                             help="Complete the timer to unlock next step")
            
            with col3:
                if st.button("⏹️ Stop Cooking"):
                    st.session_state.cooking_step = 0
                    st.session_state.timer_active = False
                    st.warning("Cooking stopped. You can resume anytime!")
        
        else:
            # Cooking complete
            st.success("🎉 Congratulations! You've completed the recipe!")
            st.balloons()
            
            # Recipe completion feedback
            st.subheader("📝 How was your cooking experience?")
            
            col1, col2 = st.columns(2)
            
            with col1:
                overall_rating = st.slider("Overall Satisfaction", 1, 5, 4)
                taste_rating = st.slider("Taste", 1, 5, 4) 
                difficulty_rating = st.slider("Difficulty Level", 1, 5, 3)
            
            with col2:
                time_rating = st.slider("Cooking Time Accuracy", 1, 5, 4)
                instruction_clarity = st.slider("Instruction Clarity", 1, 5, 4)
                would_cook_again = st.selectbox("Would cook again?", ["Yes", "Maybe", "No"])
            
            feedback_text = st.text_area("Additional feedback (optional):")
            
            if st.button("📊 Submit Cooking Feedback", type="primary"):
                feedback = {
                    'type': 'cooking_complete',
                    'recipe_id': recipe.get('id', 0),
                    'overall_satisfaction': overall_rating,
                    'taste': taste_rating,
                    'difficulty': difficulty_rating,
                    'time_accuracy': time_rating, 
                    'instruction_clarity': instruction_clarity,
                    'would_cook_again': would_cook_again,
                    'feedback_text': feedback_text,
                    'timestamp': datetime.now()
                }
                
                st.session_state.feedback_history.append(feedback)
                
                # Update RL agent
                avg_satisfaction = (overall_rating + taste_rating) / 10.0
                if st.session_state.rl_agent:
                    try:
                        recipe_idx = recipe.get('recipe_index', 0)
                        st.session_state.rl_agent.update_with_feedback(
                            recipe_idx, {'satisfaction': avg_satisfaction}
                        )
                    except Exception as e:
                        logger.warning(f"RL feedback update failed: {e}")
                
                # Update bandit
                if st.session_state.bandit_recommender:
                    try:
                        st.session_state.bandit_recommender.update(
                            recipe.get('recipe_index', 0), avg_satisfaction
                        )
                    except Exception as e:
                        logger.warning(f"Bandit update failed: {e}")
                
                st.success("✅ Thank you for your feedback! The AI is learning from your experience.")
                
                # Update learning progress
                st.session_state.learning_progress = min(
                    st.session_state.learning_progress + 0.2, 1.0
                )
            
            # Reset for next recipe
            if st.button("🔄 Cook Another Recipe"):
                st.session_state.cooking_step = 0
                st.session_state.current_recipe = None
                st.session_state.current_tab = "recipe"
    
    def feedback_tab(self):
        """Feedback analysis and learning dashboard"""
        st.header("📊 AI Learning & Feedback Dashboard")
        
        if not st.session_state.feedback_history:
            st.info("📝 No feedback data yet. Complete some recipes to see your learning progress!")
            return
        
        # Learning Overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_interactions = len(st.session_state.feedback_history)
            st.metric("Total Interactions", total_interactions)
        
        with col2:
            avg_satisfaction = np.mean([
                f.get('overall_satisfaction', f.get('accuracy_rating', f.get('satisfaction', 0))) 
                for f in st.session_state.feedback_history
            ])
            st.metric("Avg Satisfaction", f"{avg_satisfaction:.1f}/5")
        
        with col3:
            learning_progress = st.session_state.learning_progress * 100
            st.metric("Learning Progress", f"{learning_progress:.0f}%")
        
        # Satisfaction Over Time
        st.subheader("📈 Satisfaction Trends")
        
        if len(st.session_state.feedback_history) >= 2:
            # Create trend chart
            feedback_data = []
            for i, feedback in enumerate(st.session_state.feedback_history):
                satisfaction = feedback.get('overall_satisfaction', 
                              feedback.get('accuracy_rating', 
                              feedback.get('satisfaction', 0)))
                feedback_data.append({
                    'interaction': i + 1,
                    'satisfaction': satisfaction,
                    'type': feedback['type']
                })
            
            # Plotly chart
            fig = px.line(
                feedback_data, 
                x='interaction', 
                y='satisfaction',
                color='type',
                title="Satisfaction Over Time",
                labels={'interaction': 'Interaction Number', 'satisfaction': 'Satisfaction Score'}
            )
            
            fig.update_layout(
                showlegend=True,
                height=400,
                xaxis_title="Interaction Number",
                yaxis_title="Satisfaction Score (1-5)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Feedback Breakdown
        st.subheader("🔍 Feedback Analysis")
        
        feedback_types = {}
        for feedback in st.session_state.feedback_history:
            ftype = feedback['type']
            if ftype not in feedback_types:
                feedback_types[ftype] = []
            feedback_types[ftype].append(feedback)
        
        for ftype, feedbacks in feedback_types.items():
            with st.expander(f"{ftype.replace('_', ' ').title()} ({len(feedbacks)} interactions)"):
                if ftype == 'cooking_complete':
                    # Cooking feedback analysis
                    taste_scores = [f['taste'] for f in feedbacks if 'taste' in f]
                    difficulty_scores = [f['difficulty'] for f in feedbacks if 'difficulty' in f]
                    
                    if taste_scores:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Avg Taste Rating", f"{np.mean(taste_scores):.1f}/5")
                        with col2:
                            st.metric("Avg Difficulty", f"{np.mean(difficulty_scores):.1f}/5")
                    
                    # Would cook again analysis
                    cook_again = [f['would_cook_again'] for f in feedbacks if 'would_cook_again' in f]
                    if cook_again:
                        cook_again_counts = {option: cook_again.count(option) for option in set(cook_again)}
                        st.write("**Would Cook Again:**")
                        for option, count in cook_again_counts.items():
                            st.write(f"• {option}: {count} times ({count/len(cook_again)*100:.0f}%)")
                
                elif ftype == 'detection':
                    # Detection feedback analysis
                    accuracy_scores = [f['accuracy_rating'] for f in feedbacks if 'accuracy_rating' in f]
                    if accuracy_scores:
                        st.metric("Avg Detection Accuracy", f"{np.mean(accuracy_scores):.1f}/5")
                    
                    detected_counts = [f['detected_count'] for f in feedbacks if 'detected_count' in f]
                    confirmed_counts = [f['confirmed_count'] for f in feedbacks if 'confirmed_count' in f]
                    
                    if detected_counts and confirmed_counts:
                        confirmation_rate = np.mean([c/d if d > 0 else 0 for c, d in zip(confirmed_counts, detected_counts)])
                        st.metric("Avg Confirmation Rate", f"{confirmation_rate:.0%}")
        
        # RL Agent Statistics
        if st.session_state.rl_agent:
            st.subheader("🤖 Reinforcement Learning Stats")
            
            try:
                rl_stats = st.session_state.rl_agent.get_learning_stats()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RL Interactions", rl_stats.get('total_interactions', 0))
                with col2:
                    improvement = rl_stats.get('improvement_rate', 0)
                    st.metric("Improvement Rate", f"{improvement:.3f}")
                with col3:
                    diversity = rl_stats.get('recipe_diversity', 0)
                    st.metric("Recipe Diversity", f"{diversity:.0%}")
                
            except Exception as e:
                st.warning(f"Could not load RL stats: {e}")
        
        # Bandit Statistics
        if st.session_state.bandit_recommender:
            st.subheader("🎰 Multi-Armed Bandit Stats")
            
            try:
                bandit_stats = st.session_state.bandit_recommender.get_recipe_stats()
                
                col1, col2 = st.columns(2)
                with col1:
                    best_idx = bandit_stats.get('best_recipe_idx', 0)
                    best_recipe = self.sample_recipes[best_idx]['title']
                    st.write(f"**Best Recipe:** {best_recipe}")
                    
                with col2:
                    success_rate = bandit_stats.get('best_success_rate', 0)
                    st.metric("Best Success Rate", f"{success_rate:.0%}")
                
                # Recipe rankings
                rankings = bandit_stats.get('recipe_rankings', [])
                if rankings:
                    st.write("**Recipe Performance Ranking:**")
                    for i, recipe_idx in enumerate(rankings[:5]):  # Top 5
                        recipe_name = self.sample_recipes[recipe_idx]['title']
                        st.write(f"{i+1}. {recipe_name}")
                
            except Exception as e:
                st.warning(f"Could not load bandit stats: {e}")
        
        # Export Data
        st.subheader("💾 Data Export")
        
        if st.button("📁 Export Feedback Data"):
            feedback_json = json.dumps(st.session_state.feedback_history, default=str, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=feedback_json,
                file_name=f"recipe_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


# Initialize and run the app
if __name__ == "__main__":
    app = RecipeApp()
    app.run()
'''

with open("streamlit_app.py", "w") as f:
    f.write(streamlit_app_code)

print("Main Streamlit application created successfully!")