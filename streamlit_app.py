
import streamlit as st
from PIL import Image
import time
import json
import os
import io
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import OpenRouter client
try:
    import openrouter_client
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="Multi-Sensory Recipe Generator",
    page_icon="🍳",
    layout="centered"
)

# --- REFINED MODERN DARK CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background: #09090b;
        color: #e4e4e7;
    }
    .stApp { 
        background: radial-gradient(circle at 50% 50%, #181822 0%, #09090b 100%);
        background-attachment: fixed;
    }

    /* Heading Styling */
    .app-title {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        letter-spacing: -2px;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    .app-subtitle {
        font-size: 1rem;
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }

    /* Compact Layout */
    .compact-card {
        background: #18181b;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #27272a;
    }

    h1, h2, h3 { color: #ffffff !important; }
    
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        background: #ffffff;
        color: #000;
        font-weight: 600;
        border: none;
        padding: 0.5rem;
    }
    
    /* Vibrant Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff6b35, #f7931e);
        height: 10px;
        border-radius: 5px;
    }

    /* Floating Effect */
    .compact-card {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .compact-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Title with vibrant flair
    st.markdown('<div class="app-title">Multi-Sensory Recipe Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Crafting Culinary Experiences</div>', unsafe_allow_html=True)

    # State Initialization
    if 'stage' not in st.session_state: st.session_state.stage = "scan"
    
    # --- STAGE PROGRESS ---
    progress = {"scan": 0.25, "tune": 0.5, "recipe": 0.75, "cook": 1.0}.get(st.session_state.stage, 0)
    st.progress(progress)
    
    if st.session_state.stage == "scan":
        render_scan()
    elif st.session_state.stage == "tune":
        render_tune()
    elif st.session_state.stage == "recipe":
        render_recipe()
    elif st.session_state.stage == "cook":
        render_cook()

def render_scan():
    st.markdown("### 1. Scan Ingredients")
    upload = st.file_uploader("Drop image", type=['jpg', 'png'], label_visibility="collapsed")
    
    if upload:
        img = Image.open(upload)
        # Resize for compact preview
        st.image(img, width=200)
        if st.button("Analyze"):
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            if OPENROUTER_AVAILABLE:
                st.session_state.ingredients = openrouter_client.analyze_image_for_ingredients(img_bytes.getvalue())
    
    if st.session_state.get('ingredients'):
        confirmed = []
        for i, item in enumerate(st.session_state.ingredients):
            if st.checkbox(item['ingredient'].title(), value=True, key=f"ing_{i}"):
                confirmed.append(item['ingredient'])
        if st.button("Proceed"):
            st.session_state.confirmed_list = confirmed
            st.session_state.stage = "tune"
            st.rerun()

def render_tune():
    st.markdown("### 2. Flavor & Cuisine Profile")
    cuisine = st.selectbox("Choose Cuisine", ["Indian", "Italian", "Mexican", "Japanese", "Thai", "Mediterranean"])
    flavour = st.multiselect("Select Flavors", ["Spicy", "Sweet", "Savory", "Tangy", "Bitter", "Smoky", "Earthy"])
    texture = st.multiselect("Select Textures", ["Crispy", "Creamy", "Soft", "Crunchy", "Chewy", "Tender"])
    
    diff = st.select_slider("Difficulty", ["Easy", "Medium", "Hard"])
    time_max = st.slider("Max Time (mins)", 15, 60, 30)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back"): st.session_state.stage = "scan"; st.rerun()
    with col2:
        if st.button("Generate"):
            prefs = {"cuisine": cuisine, "flavour": flavour, "texture": texture, "difficulty": diff, "time": time_max}
            with st.spinner("Generating..."):
                st.session_state.recipe = openrouter_client.generate_recipe(st.session_state.confirmed_list, prefs)
                st.session_state.stage = "recipe"
                st.rerun()

def render_recipe():
    st.markdown("### 3. Your Recipe")
    r = st.session_state.recipe
    st.subheader(r['title'])
    st.write(r['description'])
    
    if st.button("Start Cooking"):
        st.session_state.stage = "cook"
        st.session_state.step_idx = 0
        st.rerun()

def render_cook():
    steps = st.session_state.recipe['steps']
    idx = st.session_state.step_idx
    
    if idx < len(steps):
        st.markdown(f"**Step {idx+1} of {len(steps)}**")
        st.info(steps[idx]['instruction'])
        
        if steps[idx].get('timing', 0) > 0:
            render_timer(steps[idx]['timing'], idx)
        
        if st.button("Next Step"):
            st.session_state.step_idx += 1
            st.rerun()
    else:
        st.success("Congratulations! Meal Ready.")
        if st.button("Done"): st.session_state.stage = "scan"; st.rerun()

def render_timer(mins, step_idx):
    # Reset timer if we've moved to a new step
    if st.session_state.get('last_step_idx') != step_idx:
        st.session_state.timer_rem = mins * 60
        st.session_state.timer_active = False
        st.session_state.last_step_idx = step_idx
        st.session_state.last_time = time.time()

    # Timer logic
    if st.session_state.timer_active:
        now = time.time()
        elapsed = now - st.session_state.last_time
        st.session_state.timer_rem -= elapsed
        st.session_state.last_time = now
        
        if st.session_state.timer_rem <= 0:
            st.session_state.timer_rem = 0
            st.session_state.timer_active = False
            st.toast("Time's up!")

    # Display
    rem = max(0, int(st.session_state.timer_rem))
    st.write(f"### ⏱️ {rem // 60:02}:{rem % 60:02}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start/Pause", key=f"btn_timer_{step_idx}"):
            st.session_state.timer_active = not st.session_state.timer_active
            st.session_state.last_time = time.time()
            st.rerun()
    with col2:
        if st.button("Reset", key=f"btn_reset_{step_idx}"):
            st.session_state.timer_rem = mins * 60
            st.session_state.timer_active = False
            st.rerun()

    if st.session_state.timer_active:
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()
