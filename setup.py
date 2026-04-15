#!/usr/bin/env python3
"""
Updated Setup script for AI-Powered Multi-Sensory Recipe Generator
Compatible with Python 3.13 and latest package versions
"""

import subprocess
import sys
import os
from pathlib import Path
import platform

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False

    if version.minor >= 13:
        print("✅ Python 3.13 detected - using latest package versions")

    return True

def install_pytorch():
    """Install PyTorch with appropriate index URL"""
    print("🔥 Installing PyTorch...")
    try:
        # For Windows and Python 3.13, use CPU-only PyTorch initially
        if platform.system() == "Windows":
            cmd = [
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", 
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ]
        else:
            cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision"]

        subprocess.check_call(cmd)
        print("✅ PyTorch installed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"⚠️ PyTorch installation failed: {e}")
        print("Trying alternative installation...")

        try:
            # Fallback to latest available version
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "--upgrade"
            ])
            print("✅ PyTorch installed with fallback method!")
            return True
        except Exception as e2:
            print(f"❌ PyTorch installation failed completely: {e2}")
            return False

def install_core_packages():
    """Install core ML packages"""
    print("🤖 Installing core AI packages...")

    core_packages = [
        "transformers>=4.40.0",
        "sentence-transformers>=3.0.0", 
        "accelerate>=0.30.0"
    ]

    for package in core_packages:
        try:
            print(f"  Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to install {package}: {e}")
            # Try without version constraint
            pkg_name = package.split(">=")[0] 
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", pkg_name
                ])
                print(f"✅ Installed {pkg_name} without version constraint")
            except Exception as e2:
                print(f"❌ Could not install {pkg_name}: {e2}")

    return True

def install_rl_packages():
    """Install reinforcement learning packages"""
    print("🎯 Installing reinforcement learning packages...")

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "stable-baselines3>=2.3.0", "gymnasium>=0.29.0"
        ])
        print("✅ RL packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ RL packages installation failed: {e}")
        # Try individual installation
        for pkg in ["stable-baselines3", "gymnasium"]:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                print(f"✅ Installed {pkg}")
            except Exception as e2:
                print(f"❌ Failed to install {pkg}: {e2}")
        return True

def install_web_packages():
    """Install web framework and utilities"""
    print("🌐 Installing web framework...")

    web_packages = [
        "streamlit>=1.35.0",
        "plotly>=5.17.0", 
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "requests>=2.31.0"
    ]

    for package in web_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            # Try without version constraint
            pkg_name = package.split(">=")[0]
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
                print(f"✅ Installed {pkg_name}")
            except Exception as e:
                print(f"❌ Could not install {pkg_name}: {e}")

    return True

def install_optional_packages():
    """Install optional packages for enhanced functionality"""
    print("📦 Installing optional packages...")

    optional = [
        "opencv-python",
        "scikit-learn", 
        "matplotlib",
        "seaborn",
        "python-dotenv"
    ]

    for package in optional:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  ✅ {package}")
        except subprocess.CalledProcessError:
            print(f"  ⚠️ {package} (optional - skipped)")

def test_imports():
    """Test if critical packages can be imported"""
    print("🧪 Testing package imports...")

    test_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face Transformers"),
        ("streamlit", "Streamlit"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow")
    ]

    success_count = 0

    for package, name in test_packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
            success_count += 1
        except ImportError:
            print(f"  ❌ {name}")

    if success_count >= 4:  # At least core packages work
        print(f"✅ {success_count}/{len(test_packages)} packages working!")
        return True
    else:
        print(f"⚠️ Only {success_count}/{len(test_packages)} packages working")
        return False

def download_models():
    """Download AI models (optional)"""
    print("🤖 Attempting to download AI models...")

    try:
        # Test if we can import the packages first
        import transformers
        from sentence_transformers import SentenceTransformer

        print("  📸 Downloading BLIP model...")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        print("  🔤 Downloading sentence transformer...")
        SentenceTransformer('all-MiniLM-L6-v2')

        print("✅ Models downloaded successfully!")
        return True

    except Exception as e:
        print(f"⚠️ Model download failed: {e}")
        print("Models will be downloaded automatically on first use.")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")

    directories = [
        "models",
        "data", 
        "logs",
        "tensorboard_logs",
        "user_data",
        "cache"
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

    print("✅ Directories created!")

def create_simple_test():
    """Create a simple test script"""
    test_script = """
# Simple test to verify installation
print("🧪 Testing AI Recipe Generator Installation...")

try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
except ImportError:
    print("❌ PyTorch not available")

try:
    import transformers
    print(f"✅ Transformers {transformers.__version__}")
except ImportError:
    print("❌ Transformers not available")

try:
    import streamlit
    print(f"✅ Streamlit {streamlit.__version__}")
except ImportError:
    print("❌ Streamlit not available")

try:
    import stable_baselines3
    print(f"✅ Stable-Baselines3 {stable_baselines3.__version__}")
except ImportError:
    print("❌ Stable-Baselines3 not available")

print("\n🎉 Installation test complete!")
print("If you see ✅ for most packages, you're ready to go!")
"""

    with open("test_installation.py", "w") as f:
        f.write(test_script)

    print("✅ Created test_installation.py")

def main():
    """Main setup function"""
    print("🍳 AI Recipe Generator Setup (Updated for Python 3.13)")
    print("=" * 60)

    # Check Python version
    if not check_python_version():
        return

    # Create directories
    create_directories()

    # Install packages step by step
    print("\n📦 Installing packages in stages...")

    success = True

    # Stage 1: PyTorch
    if not install_pytorch():
        success = False

    # Stage 2: Core AI packages
    install_core_packages()

    # Stage 3: RL packages  
    install_rl_packages()

    # Stage 4: Web packages
    install_web_packages()

    # Stage 5: Optional packages
    install_optional_packages()

    # Test imports
    if test_imports():
        print("\n✅ Core installation successful!")

        # Try to download models
        download_models()

        # Create test file
        create_simple_test()

        print("\n🎉 Setup completed!")
        print("\n🚀 Next steps:")
        print("1. Test installation: python test_installation.py") 
        print("2. Run the app: python run.py")
        print("3. Or directly: streamlit run streamlit_app.py")

    else:
        print("\n⚠️ Some packages failed to install.")
        print("\n🔧 Manual installation options:")
        print("1. Install PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("2. Install Transformers: pip install transformers sentence-transformers")
        print("3. Install Streamlit: pip install streamlit")
        print("4. Install RL: pip install stable-baselines3 gymnasium")
        print("\nThen test with: python test_installation.py")

if __name__ == "__main__":
    main()
