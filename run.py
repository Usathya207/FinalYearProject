#!/usr/bin/env python3
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
        import requests
        import dotenv
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
        print("\nPress Ctrl+C to stop the application")

        # Wait for process
        process.wait()

    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
        process.terminate()
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    """Main run function"""
    print("🍳 AI-Powered Multi-Sensory Recipe Generator")
    print("=" * 50)

    # Check if setup was run
    req_path = Path(__file__).parent / "requirements.txt"
    if not req_path.exists():
        print(f"❌ Requirements file not found at {req_path}. Please check your files.")
        return

    # Check requirements
    if not check_requirements():
        print("❌ Requirements not installed. Running setup...")
        subprocess.check_call([sys.executable, "setup.py"])

    # Run application
    run_streamlit()

if __name__ == "__main__":
    main()
