# Create setup script and run instructions (fixed version)
setup_script = '''#!/usr/bin/env python3
"""
Setup script for AI-Powered Multi-Sensory Recipe Generator
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def download_models():
    """Download and cache Hugging Face models"""
    print("🤖 Downloading AI models...")
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from sentence_transformers import SentenceTransformer
        
        # Download BLIP model for ingredient detection
        print("  📸 Downloading BLIP model for ingredient detection...")
        BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Download sentence transformer
        print("  🔤 Downloading sentence transformer...")
        SentenceTransformer('all-MiniLM-L6-v2')
        
        print("✅ Models downloaded and cached successfully!")
        
    except Exception as e:
        print(f"⚠️ Model download failed: {e}")
        print("Models will be downloaded on first use.")
    
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "models",
        "data", 
        "logs",
        "tensorboard_logs",
        "user_data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Directories created successfully!")

def setup_environment():
    """Setup environment variables and configuration"""
    print("🌍 Setting up environment...")
    
    env_content = """# AI Recipe Generator Environment Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
HUGGINGFACE_HUB_CACHE=./models/huggingface_cache
PYTORCH_TRANSFORMERS_CACHE=./models/pytorch_cache
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Environment configured successfully!")

def main():
    """Main setup function"""
    print("🍳 AI-Powered Multi-Sensory Recipe Generator Setup")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Setup environment
    setup_environment()
    
    # Install requirements
    if not install_requirements():
        return
    
    # Download models
    download_models()
    
    print("\\n🎉 Setup completed successfully!")
    print("\\nTo run the application:")
    print("  streamlit run streamlit_app.py")
    print("\\nOr use the run script:")
    print("  python run.py")

if __name__ == "__main__":
    main()
'''

with open("setup.py", "w") as f:
    f.write(setup_script)

print("Setup script created successfully!")