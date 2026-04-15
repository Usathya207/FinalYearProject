# Create setup script and run instructions
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
    
    env_content = '''
# AI Recipe Generator Environment Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
HUGGINGFACE_HUB_CACHE=./models/huggingface_cache
PYTORCH_TRANSFORMERS_CACHE=./models/pytorch_cache
'''
    
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

# Create run script
run_script = '''#!/usr/bin/env python3
"""
Run script for AI-Powered Multi-Sensory Recipe Generator
"""

import subprocess
import sys
import os
from pathlib import Path
import webbrowser
import time

def check_requirements():
    """Check if requirements are installed"""
    try:
        import streamlit
        import torch
        import transformers
        return True
    except ImportError:
        return False

def run_streamlit():
    """Run the Streamlit application"""
    print("🚀 Starting AI Recipe Generator...")
    
    # Set environment variables
    env = os.environ.copy()
    env['STREAMLIT_SERVER_PORT'] = '8501'
    env['STREAMLIT_SERVER_ADDRESS'] = 'localhost'
    
    try:
        # Start Streamlit
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false"
        ], env=env)
        
        # Wait a moment and open browser
        time.sleep(3)
        webbrowser.open("http://localhost:8501")
        
        print("✅ Application started successfully!")
        print("🌐 Opening in your web browser...")
        print("📱 Access at: http://localhost:8501")
        print("\\nPress Ctrl+C to stop the application")
        
        # Wait for process
        process.wait()
        
    except KeyboardInterrupt:
        print("\\n🛑 Application stopped by user")
        process.terminate()
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    """Main run function"""
    print("🍳 AI-Powered Multi-Sensory Recipe Generator")
    print("=" * 50)
    
    # Check if setup was run
    if not Path("requirements.txt").exists():
        print("❌ Requirements file not found. Please run setup.py first!")
        return
    
    # Check requirements
    if not check_requirements():
        print("❌ Requirements not installed. Running setup...")
        subprocess.check_call([sys.executable, "setup.py"])
    
    # Run application
    run_streamlit()

if __name__ == "__main__":
    main()
'''

with open("run.py", "w") as f:
    f.write(run_script)

# Make scripts executable
import stat
for script in ["setup.py", "run.py"]:
    if os.path.exists(script):
        st = os.stat(script)
        os.chmod(script, st.st_mode | stat.S_IEXEC)

print("Setup and run scripts created successfully!")