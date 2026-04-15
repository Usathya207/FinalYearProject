
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

print("
🎉 Installation test complete!")
print("If you see ✅ for most packages, you're ready to go!")
