import os
import re
import time
import streamlit as st
import pandas as pd
from PIL import Image
from transformers import pipeline

DATASET_FOLDER = "./data"


def load_and_standardize(filepath, columns_map):
    df = pd.read_csv(filepath, encoding="utf-8")
    df = df.rename(columns=columns_map)
    cols = ['title', 'ingredients', 'instructions', 'image']
    existing_cols = [col for col in cols if col in df.columns]
    return df[existing_cols]


dataset_files = {
    "cleaned_indian_recipes": os.path.join(DATASET_FOLDER, "Cleaned_Indian_Food_Dataset.csv"),
}

columns_mappings = {
    "cleaned_indian_recipes": {
        "TranslatedRecipeName": "title",
        "Cleaned-Ingredients": "ingredients",
        "TranslatedInstructions": "instructions",
        "image-url": "image",
    }
}


def prepare_dataset():
    dfs = []
    for key, path in dataset_files.items():
        if os.path.exists(path):
            dfs.append(load_and_standardize(path, columns_mappings.get(key, {})))
        else:
            st.warning(f"Dataset file not found: {path}")
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        return combined
    return pd.DataFrame()


recipes_df = prepare_dataset()


@st.cache_resource(show_spinner=False)
def get_hf_pipeline():
    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")


classifier = get_hf_pipeline()

candidate_ingredients = [
    "tomato", "onion", "garlic", "chicken", "pepper", "potato", "ginger", "coriander", "curry leaves", "chili",
    "lemon", "oil", "cumin", "turmeric", "cardamom", "cinnamon", "cloves", "mustard seeds", "fennel", "green chili"
]


def detect_ingredients(image):
    predictions = classifier(image, candidate_ingredients)
    detected = [item['label'] for item in predictions if item['score'] > 0.1]
    return detected


def matches_preferences(row, ing_set, flavor_pref, texture_pref):
    recipe_ingredients = set(ing.strip().lower() for ing in row.get('ingredients', "").split(","))
    if not ing_set.issubset(recipe_ingredients):
        return False
    flavor_pass = flavor_pref.lower() in row['title'].lower() or flavor_pref.lower() in row.get('instructions',
                                                                                                '').lower()
    texture_pass = texture_pref.lower() in row['title'].lower() or texture_pref.lower() in row.get('instructions',
                                                                                                   '').lower()
    if flavor_pref == "Any":
        flavor_pass = True
    if texture_pref == "Any":
        texture_pass = True
    return flavor_pass and texture_pass


def search_recipes(recipes_df, detected_ingredients, flavor_pref, texture_pref):
    ing_set = set(i.lower() for i in detected_ingredients)
    matched = []
    for _, row in recipes_df.iterrows():
        if matches_preferences(row, ing_set, flavor_pref, texture_pref):
            matched.append(row)
    return pd.DataFrame(matched)


def parse_timer(text):
    match = re.search(r'(\d+)\s*(seconds|minutes|min|secs|sec)', text.lower())
    if match:
        num = int(match.group(1))
        unit = match.group(2)
        if 'minute' in unit or 'min' in unit:
            return num * 60
        elif 'second' in unit or 'sec' in unit:
            return num
    return None


# Session state defaults
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'timer_running' not in st.session_state:
    st.session_state.timer_running = False
if 'timer_start' not in st.session_state:
    st.session_state.timer_start = None
if 'timer_seconds' not in st.session_state:
    st.session_state.timer_seconds = 0
if 'rewards' not in st.session_state:
    st.session_state.rewards = 0
if 'recipe_steps' not in st.session_state:
    st.session_state.recipe_steps = []
if 'confirm_index' not in st.session_state:
    st.session_state.confirm_index = 0
if 'confirm_done' not in st.session_state:
    st.session_state.confirm_done = False
if 'selected_ingredients' not in st.session_state:
    st.session_state.selected_ingredients = []
if 'detected_ingredients' not in st.session_state:
    st.session_state.detected_ingredients = []

st.title("🍳 Multi-Sensory Recipe Assistant (No Rerun)")

uploaded_file = st.file_uploader("Upload an image of your ingredients", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Ingredients Image", use_container_width=True)

    if st.button("Detect Ingredients"):
        detected_ingredients = detect_ingredients(image)
        st.session_state.detected_ingredients = detected_ingredients
        st.session_state.confirm_index = 0
        st.session_state.selected_ingredients = []
        st.session_state.rewards = 0
        st.session_state.confirm_done = False

    if not st.session_state.confirm_done and st.session_state.detected_ingredients:
        idx = st.session_state.confirm_index
        if idx < len(st.session_state.detected_ingredients):
            current_ingr = st.session_state.detected_ingredients[idx]
            st.write(f"Is '{current_ingr}' a correct ingredient?")
            confirm = st.radio("", ('Yes', 'No'), key=f"confirm_{idx}")
            if st.button("Next"):
                if confirm == "Yes":
                    st.session_state.selected_ingredients.append(current_ingr)
                    st.session_state.rewards += 1
                else:
                    st.session_state.rewards -= 1
                st.session_state.confirm_index += 1

        st.write(f"Reward score: {st.session_state.rewards}")

        if st.session_state.confirm_index >= len(st.session_state.detected_ingredients):
            st.session_state.confirm_done = True

    elif st.session_state.confirm_done:
        st.success(f"Confirmed ingredients: {', '.join(st.session_state.selected_ingredients)}")

        flavor = st.selectbox("Select your preferred flavor profile", ['Any', 'Spicy', 'Sweet', 'Tangy', 'Savory'])
        texture = st.selectbox("Select your preferred texture", ['Any', 'Crispy', 'Soggy', 'Tender', 'Crunchy'])

        if st.button("Find Matching Recipes"):
            matched = search_recipes(recipes_df, st.session_state.selected_ingredients, flavor, texture)
            if matched.empty:
                st.info("No matching recipes found with your preferences.")
                st.session_state.recipe_steps = []
                st.session_state.step = 0
            else:
                st.success(f"Found {len(matched)} matching recipes.")
                st.session_state.matched_recipes = matched
                st.session_state.current_recipe_idx = 0
                st.session_state.active_recipe = matched.iloc[0]
                instructions_raw = st.session_state.active_recipe['instructions']
                steps = re.split(r'\.\s+|\n|!\s+', instructions_raw)
                st.session_state.recipe_steps = [step.strip() for step in steps if step.strip()]
                st.session_state.step = 0

if st.session_state.recipe_steps:
    step = st.session_state.step
    current_step_text = st.session_state.recipe_steps[step]

    st.markdown(f"### Step {step + 1} of {len(st.session_state.recipe_steps)}")
    st.write(current_step_text)

    if not st.session_state.timer_running:
        seconds = parse_timer(current_step_text)
        st.session_state.timer_seconds = seconds if seconds else 0
        st.session_state.timer_start = None

    if st.session_state.timer_seconds > 0:
        if st.button("Start Timer"):
            st.session_state.timer_running = True
            st.session_state.timer_start = time.time()

        if st.session_state.timer_running and st.session_state.timer_start is not None:
            elapsed = time.time() - st.session_state.timer_start
            remaining = st.session_state.timer_seconds - elapsed
            if remaining > 0:
                st.info(f"Time left: {int(remaining)} seconds")
            else:
                st.success("Timer finished!")
                st.session_state.timer_running = False

        if st.button("Skip Timer"):
            st.session_state.timer_running = False

    else:
        st.info("No timer needed for this step")

    if not st.session_state.timer_running:
        if step + 1 < len(st.session_state.recipe_steps):
            if st.button("Next Step"):
                st.session_state.step += 1
                st.session_state.timer_running = False
                st.session_state.timer_start = None
        else:
            st.success("You have completed all the steps!")

st.sidebar.header("Session Info")
st.sidebar.write(f"Reward score: {st.session_state.rewards}")
if st.session_state.selected_ingredients:
    st.sidebar.write(f"Selected Ingredients: {', '.join(st.session_state.selected_ingredients)}")
if 'active_recipe' in st.session_state:
    st.sidebar.write(f"Active Recipe: {st.session_state.active_recipe['title']}")
    st.sidebar.write(f"Total Steps: {len(st.session_state.recipe_steps)}")

st.sidebar.header("Dataset Info")
st.sidebar.write(f"Loaded {len(recipes_df)} recipes.")
