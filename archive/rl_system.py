
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import pickle
import json
from typing import Dict, List, Tuple, Any
from collections import deque, defaultdict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class RecipeRecommendationEnv(gym.Env):
    """
    Custom Gymnasium environment for recipe recommendation using reinforcement learning
    """

    def __init__(self, recipe_database: List[Dict], max_episodes: int = 1000):
        super().__init__()

        self.recipe_database = recipe_database
        self.max_episodes = max_episodes
        self.current_episode = 0

        # Define action space (recipe selection indices)
        self.action_space = spaces.Discrete(len(recipe_database))

        # Define observation space (user preferences + historical interactions)
        # [flavor_preferences(5), texture_preferences(6), dietary_restrictions(7), 
        #  cooking_time_pref(1), serving_size_pref(1), satisfaction_history(10)]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(30,), dtype=np.float32
        )

        # User state tracking
        self.user_preferences = {
            'flavor': np.zeros(5),  # spicy, sweet, tangy, savory, bitter
            'texture': np.zeros(6),  # crispy, soft, chewy, tender, crunchy, creamy
            'dietary': np.zeros(7),  # various dietary restrictions
            'cooking_time': 0.5,    # normalized cooking time preference
            'serving_size': 0.5     # normalized serving size preference
        }

        # Interaction history
        self.satisfaction_history = deque(maxlen=10)
        self.recipe_history = deque(maxlen=50)
        self.interaction_count = 0

        # Reward parameters
        self.satisfaction_weight = 1.0
        self.diversity_weight = 0.3
        self.novelty_weight = 0.2

    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)

        # Initialize random user preferences
        if self.current_episode == 0:
            self._initialize_user_preferences()

        self.current_episode += 1
        return self._get_observation(), {}

    def step(self, action: int):
        """Execute action and return results"""
        try:
            # Get selected recipe
            recipe = self.recipe_database[action]

            # Simulate user interaction and get feedback
            satisfaction_score = self._simulate_user_feedback(recipe)

            # Calculate reward
            reward = self._calculate_reward(recipe, satisfaction_score)

            # Update history
            self.satisfaction_history.append(satisfaction_score)
            self.recipe_history.append(action)
            self.interaction_count += 1

            # Check if episode is done
            done = self.interaction_count >= 100  # Max interactions per episode

            # Get next observation
            obs = self._get_observation()

            # Additional info
            info = {
                'recipe_id': action,
                'satisfaction': satisfaction_score,
                'recipe_title': recipe.get('title', 'Unknown'),
                'reward_breakdown': {
                    'satisfaction': satisfaction_score * self.satisfaction_weight,
                    'diversity': self._calculate_diversity_bonus(action) * self.diversity_weight,
                    'novelty': self._calculate_novelty_bonus(action) * self.novelty_weight
                }
            }

            return obs, reward, done, False, info

        except Exception as e:
            logger.error(f"Error in RL step: {e}")
            return self._get_observation(), -1.0, True, False, {}

    def _initialize_user_preferences(self):
        """Initialize user preferences with some randomness to simulate different users"""
        # Random flavor preferences
        flavor_weights = np.random.dirichlet([1, 1, 1, 1, 1])
        self.user_preferences['flavor'] = flavor_weights

        # Random texture preferences
        texture_prefs = np.random.random(6)
        texture_prefs = texture_prefs / np.sum(texture_prefs)
        self.user_preferences['texture'] = texture_prefs

        # Random dietary restrictions
        self.user_preferences['dietary'] = np.random.random(7) > 0.8

        # Cooking time and serving preferences
        self.user_preferences['cooking_time'] = np.random.uniform(0.2, 0.8)
        self.user_preferences['serving_size'] = np.random.uniform(0.3, 0.7)

    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        obs = np.concatenate([
            self.user_preferences['flavor'],
            self.user_preferences['texture'],
            self.user_preferences['dietary'].astype(np.float32),
            [self.user_preferences['cooking_time']],
            [self.user_preferences['serving_size']],
            list(self.satisfaction_history) + [0.0] * (10 - len(self.satisfaction_history))
        ])
        return obs.astype(np.float32)

    def _simulate_user_feedback(self, recipe: Dict) -> float:
        """Simulate realistic user feedback based on preferences"""
        satisfaction = 0.5  # Base satisfaction

        # Flavor alignment
        recipe_flavors = self._extract_recipe_flavors(recipe)
        flavor_alignment = np.dot(self.user_preferences['flavor'], recipe_flavors)
        satisfaction += 0.3 * flavor_alignment

        # Texture alignment
        recipe_textures = self._extract_recipe_textures(recipe)
        texture_alignment = np.dot(self.user_preferences['texture'], recipe_textures)
        satisfaction += 0.2 * texture_alignment

        # Cooking time preference
        cook_time_normalized = min(recipe.get('cook_time', 30) / 60.0, 1.0)
        time_diff = abs(cook_time_normalized - self.user_preferences['cooking_time'])
        satisfaction += 0.1 * (1.0 - time_diff)

        # Add some noise for realism
        satisfaction += np.random.normal(0, 0.05)

        return np.clip(satisfaction, 0.0, 1.0)

    def _extract_recipe_flavors(self, recipe: Dict) -> np.ndarray:
        """Extract flavor profile from recipe"""
        flavors = np.zeros(5)  # spicy, sweet, tangy, savory, bitter

        title = recipe.get('title', '').lower()
        description = recipe.get('description', '').lower()
        ingredients = ' '.join([ing.get('item', '') for ing in recipe.get('ingredients', [])])

        text = f"{title} {description} {ingredients}".lower()

        # Simple keyword-based flavor detection
        if any(word in text for word in ['spicy', 'hot', 'chili', 'pepper', 'sriracha']):
            flavors[0] = 1.0
        if any(word in text for word in ['sweet', 'honey', 'sugar', 'maple']):
            flavors[1] = 1.0
        if any(word in text for word in ['tangy', 'lemon', 'lime', 'vinegar', 'citrus']):
            flavors[2] = 1.0
        if any(word in text for word in ['savory', 'herbs', 'garlic', 'onion']):
            flavors[3] = 1.0
        if any(word in text for word in ['bitter', 'dark', 'coffee']):
            flavors[4] = 1.0

        # Normalize
        return flavors / (np.sum(flavors) + 1e-8)

    def _extract_recipe_textures(self, recipe: Dict) -> np.ndarray:
        """Extract texture profile from recipe"""
        textures = np.zeros(6)  # crispy, soft, chewy, tender, crunchy, creamy

        title = recipe.get('title', '').lower()
        description = recipe.get('description', '').lower()

        text = f"{title} {description}".lower()

        if 'crispy' in text or 'fried' in text: textures[0] = 1.0
        if 'soft' in text or 'tender' in text: textures[1] = 1.0
        if 'chewy' in text: textures[2] = 1.0
        if 'tender' in text: textures[3] = 1.0
        if 'crunchy' in text or 'crisp' in text: textures[4] = 1.0
        if 'creamy' in text or 'cream' in text: textures[5] = 1.0

        return textures / (np.sum(textures) + 1e-8)

    def _calculate_reward(self, recipe: Dict, satisfaction: float) -> float:
        """Calculate reward for the current action"""
        # Base reward from satisfaction
        reward = satisfaction * self.satisfaction_weight

        # Diversity bonus (encouraging variety)
        diversity_bonus = self._calculate_diversity_bonus(
            self.recipe_database.index(recipe)
        )
        reward += diversity_bonus * self.diversity_weight

        # Novelty bonus (encouraging exploration)
        novelty_bonus = self._calculate_novelty_bonus(
            self.recipe_database.index(recipe)
        )
        reward += novelty_bonus * self.novelty_weight

        return reward

    def _calculate_diversity_bonus(self, recipe_idx: int) -> float:
        """Calculate diversity bonus based on recent recipe history"""
        if len(self.recipe_history) < 5:
            return 0.5

        recent_recipes = list(self.recipe_history)[-5:]
        if recipe_idx in recent_recipes:
            return -0.2  # Penalty for repetition
        else:
            return 0.3   # Bonus for diversity

    def _calculate_novelty_bonus(self, recipe_idx: int) -> float:
        """Calculate novelty bonus for unexplored recipes"""
        recipe_count = list(self.recipe_history).count(recipe_idx)
        if recipe_count == 0:
            return 0.5  # High bonus for never-tried recipes
        elif recipe_count == 1:
            return 0.2  # Medium bonus for rarely-tried recipes
        else:
            return 0.0  # No bonus for frequently-tried recipes


class PersonalizationAgent:
    """
    Reinforcement Learning agent for personalized recipe recommendations
    """

    def __init__(self, recipe_database: List[Dict], model_type: str = "PPO"):
        self.recipe_database = recipe_database
        self.model_type = model_type

        # Create environment
        self.env = RecipeRecommendationEnv(recipe_database)

        # Initialize RL model
        if model_type == "PPO":
            self.model = PPO(
                "MlpPolicy",
                self.env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                tensorboard_log="./ppo_recipe_tensorboard/"
            )
        elif model_type == "DQN":
            self.model = DQN(
                "MlpPolicy",
                self.env,
                verbose=1,
                learning_rate=1e-4,
                buffer_size=50000,
                learning_starts=1000,
                batch_size=32,
                target_update_interval=1000,
                gamma=0.99,
                tensorboard_log="./dqn_recipe_tensorboard/"
            )

        # User feedback storage
        self.user_feedback_history = defaultdict(list)
        self.user_preferences_history = defaultdict(dict)

    def train(self, total_timesteps: int = 10000):
        """Train the RL agent"""
        logger.info(f"Training {self.model_type} agent for {total_timesteps} timesteps...")

        try:
            self.model.learn(total_timesteps=total_timesteps)
            logger.info("Training completed successfully!")
        except Exception as e:
            logger.error(f"Training failed: {e}")

    def recommend_recipe(self, user_preferences: Dict, top_k: int = 3) -> List[Dict]:
        """Get personalized recipe recommendations"""
        try:
            # Update environment with current user preferences
            self._update_environment_preferences(user_preferences)

            # Get current observation
            obs = self.env._get_observation()

            # Get action probabilities from the model
            if hasattr(self.model, 'predict_proba'):
                action_probs = self.model.predict_proba(obs)
            else:
                # For DQN, we'll use Q-values as a proxy for preferences
                with torch.no_grad():
                    q_values = self.model.q_net(torch.FloatTensor(obs).unsqueeze(0))
                    action_probs = torch.softmax(q_values, dim=1).numpy().flatten()

            # Get top-k recommendations
            top_indices = np.argsort(action_probs)[-top_k:][::-1]

            recommendations = []
            for idx in top_indices:
                recipe = self.recipe_database[idx]
                recipe['recommendation_score'] = float(action_probs[idx])
                recipe['recipe_index'] = idx
                recommendations.append(recipe)

            return recommendations

        except Exception as e:
            logger.error(f"Error in recipe recommendation: {e}")
            return self._get_fallback_recommendations(top_k)

    def update_with_feedback(self, recipe_idx: int, feedback: Dict):
        """Update the agent with user feedback"""
        try:
            # Extract satisfaction score from feedback
            satisfaction = feedback.get('satisfaction', 0.5)

            # Simulate an environment step with this feedback
            obs = self.env._get_observation()
            _, reward, _, _, info = self.env.step(recipe_idx)

            # Store feedback for analysis
            self.user_feedback_history['satisfaction'].append(satisfaction)
            self.user_feedback_history['recipes'].append(recipe_idx)
            self.user_feedback_history['timestamps'].append(datetime.now())

            # Update user preferences based on feedback
            self._adapt_preferences_from_feedback(recipe_idx, feedback)

            logger.info(f"Updated agent with feedback: satisfaction={satisfaction}, reward={reward}")

        except Exception as e:
            logger.error(f"Error updating with feedback: {e}")

    def _update_environment_preferences(self, user_preferences: Dict):
        """Update environment with current user preferences"""
        # Map user preferences to environment format
        flavor_mapping = {
            'spicy': 0, 'sweet': 1, 'tangy': 2, 'savory': 3, 'bitter': 4
        }

        texture_mapping = {
            'crispy': 0, 'soft': 1, 'chewy': 2, 'tender': 3, 'crunchy': 4, 'creamy': 5
        }

        # Update flavor preferences
        flavor = user_preferences.get('flavor')
        if flavor and flavor in flavor_mapping:
            self.env.user_preferences['flavor'].fill(0.1)  # Reset with small values
            self.env.user_preferences['flavor'][flavor_mapping[flavor]] = 1.0

        # Update texture preferences
        textures = user_preferences.get('textures', [])
        self.env.user_preferences['texture'].fill(0.1)  # Reset with small values
        for texture in textures:
            if texture in texture_mapping:
                self.env.user_preferences['texture'][texture_mapping[texture]] = 1.0

        # Update other preferences
        self.env.user_preferences['cooking_time'] = user_preferences.get('cooking_time', 30) / 60.0
        self.env.user_preferences['serving_size'] = user_preferences.get('serving_size', 4) / 8.0

    def _adapt_preferences_from_feedback(self, recipe_idx: int, feedback: Dict):
        """Adapt user preferences based on feedback"""
        recipe = self.recipe_database[recipe_idx]
        satisfaction = feedback.get('satisfaction', 0.5)

        # If user liked the recipe, strengthen preferences towards its characteristics
        if satisfaction > 0.7:
            recipe_flavors = self.env._extract_recipe_flavors(recipe)
            recipe_textures = self.env._extract_recipe_textures(recipe)

            # Strengthen preferences (simple exponential smoothing)
            alpha = 0.1
            self.env.user_preferences['flavor'] = (
                (1 - alpha) * self.env.user_preferences['flavor'] +
                alpha * recipe_flavors
            )
            self.env.user_preferences['texture'] = (
                (1 - alpha) * self.env.user_preferences['texture'] +
                alpha * recipe_textures
            )

        # If user didn't like the recipe, slightly weaken those preferences
        elif satisfaction < 0.3:
            recipe_flavors = self.env._extract_recipe_flavors(recipe)
            recipe_textures = self.env._extract_recipe_textures(recipe)

            alpha = 0.05
            self.env.user_preferences['flavor'] = (
                (1 + alpha) * self.env.user_preferences['flavor'] -
                alpha * recipe_flavors
            )
            self.env.user_preferences['texture'] = (
                (1 + alpha) * self.env.user_preferences['texture'] -
                alpha * recipe_textures
            )

    def _get_fallback_recommendations(self, top_k: int) -> List[Dict]:
        """Fallback recommendations when RL fails"""
        return self.recipe_database[:top_k]

    def save_model(self, path: str):
        """Save the trained model"""
        try:
            self.model.save(path)

            # Save additional state
            state = {
                'user_feedback_history': dict(self.user_feedback_history),
                'user_preferences_history': dict(self.user_preferences_history),
                'model_type': self.model_type
            }

            with open(f"{path}_state.pkl", 'wb') as f:
                pickle.dump(state, f)

            logger.info(f"Model and state saved to {path}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load_model(self, path: str):
        """Load a trained model"""
        try:
            if self.model_type == "PPO":
                self.model = PPO.load(path, env=self.env)
            elif self.model_type == "DQN":
                self.model = DQN.load(path, env=self.env)

            # Load additional state
            with open(f"{path}_state.pkl", 'rb') as f:
                state = pickle.load(f)
                self.user_feedback_history = defaultdict(list, state['user_feedback_history'])
                self.user_preferences_history = defaultdict(dict, state['user_preferences_history'])

            logger.info(f"Model and state loaded from {path}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def get_learning_stats(self) -> Dict:
        """Get statistics about the learning process"""
        if not self.user_feedback_history['satisfaction']:
            return {'total_interactions': 0}

        satisfactions = self.user_feedback_history['satisfaction']

        stats = {
            'total_interactions': len(satisfactions),
            'average_satisfaction': np.mean(satisfactions),
            'satisfaction_trend': np.mean(satisfactions[-10:]) if len(satisfactions) >= 10 else np.mean(satisfactions),
            'improvement_rate': (np.mean(satisfactions[-10:]) - np.mean(satisfactions[:10])) if len(satisfactions) >= 20 else 0.0,
            'unique_recipes_tried': len(set(self.user_feedback_history['recipes'])),
            'recipe_diversity': len(set(self.user_feedback_history['recipes'])) / len(self.user_feedback_history['recipes']) if self.user_feedback_history['recipes'] else 0
        }

        return stats


class MultiArmedBanditRecommender:
    """
    Alternative simpler recommendation system using Multi-Armed Bandit approach
    """

    def __init__(self, recipe_database: List[Dict]):
        self.recipe_database = recipe_database
        self.n_recipes = len(recipe_database)

        # Thompson Sampling parameters
        self.alpha = np.ones(self.n_recipes)  # Success count
        self.beta = np.ones(self.n_recipes)   # Failure count

        # Contextual features
        self.recipe_features = self._extract_recipe_features()

        # User context learning
        self.user_context_weights = np.zeros(self.recipe_features.shape[1])

    def _extract_recipe_features(self) -> np.ndarray:
        """Extract numerical features from recipes"""
        features = []

        for recipe in self.recipe_database:
            feature_vector = [
                recipe.get('cook_time', 30) / 60.0,  # Normalized cooking time
                recipe.get('servings', 4) / 8.0,      # Normalized servings
                len(recipe.get('ingredients', [])) / 20.0,  # Normalized ingredient count
                len(recipe.get('steps', [])) / 15.0,   # Normalized step count
            ]

            # Add flavor and texture features (binary)
            title_desc = f"{recipe.get('title', '')} {recipe.get('description', '')}".lower()

            # Flavor features
            feature_vector.extend([
                1.0 if 'spicy' in title_desc else 0.0,
                1.0 if 'sweet' in title_desc else 0.0,
                1.0 if any(word in title_desc for word in ['tangy', 'sour']) else 0.0,
                1.0 if 'savory' in title_desc else 0.0,
            ])

            # Texture features
            feature_vector.extend([
                1.0 if 'crispy' in title_desc else 0.0,
                1.0 if 'creamy' in title_desc else 0.0,
                1.0 if 'tender' in title_desc else 0.0,
            ])

            features.append(feature_vector)

        return np.array(features)

    def recommend(self, user_context: np.ndarray, n_recommendations: int = 3) -> List[int]:
        """Get recommendations using Thompson Sampling"""
        # Sample from Beta distributions for each recipe
        sampled_rewards = np.random.beta(self.alpha, self.beta)

        # Adjust based on user context (contextual bandit)
        if len(user_context) == len(self.user_context_weights):
            context_scores = self.recipe_features @ user_context
            sampled_rewards += 0.1 * context_scores  # Small context influence

        # Select top-k recipes
        top_indices = np.argsort(sampled_rewards)[-n_recommendations:][::-1]
        return top_indices.tolist()

    def update(self, recipe_idx: int, reward: float):
        """Update bandit parameters based on feedback"""
        if reward > 0.5:  # Positive feedback
            self.alpha[recipe_idx] += 1
        else:  # Negative feedback
            self.beta[recipe_idx] += 1

    def get_recipe_stats(self) -> Dict:
        """Get statistics about recipe performance"""
        success_rates = self.alpha / (self.alpha + self.beta)

        return {
            'best_recipe_idx': int(np.argmax(success_rates)),
            'best_success_rate': float(np.max(success_rates)),
            'average_success_rate': float(np.mean(success_rates)),
            'total_trials': int(np.sum(self.alpha + self.beta) - 2 * len(self.alpha)),
            'recipe_rankings': np.argsort(success_rates)[::-1].tolist()
        }
