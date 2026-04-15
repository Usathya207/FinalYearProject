
# 🔧 Manual Installation Guide for Python 3.13

If the automatic setup fails, follow these steps:

## Step 1: Install PyTorch (Core dependency)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Step 2: Install Hugging Face packages
```bash
pip install transformers>=4.40.0
pip install sentence-transformers>=3.0.0
```

## Step 3: Install Reinforcement Learning
```bash
pip install stable-baselines3>=2.3.0
pip install gymnasium>=0.29.0
```

## Step 4: Install Web Framework
```bash
pip install streamlit>=1.35.0
pip install plotly>=5.17.0
```

## Step 5: Install Utilities
```bash
pip install Pillow numpy pandas requests
```

## Step 6: Test Installation
```bash
python test_installation.py
```

## Step 7: Run the App
```bash
python run.py
```

## Minimal Version (If still having issues):
```bash
pip install torch transformers streamlit numpy pandas requests pillow
```

Then just run:
```bash
streamlit run streamlit_app.py
```

The app will work with basic functionality even without all packages!
