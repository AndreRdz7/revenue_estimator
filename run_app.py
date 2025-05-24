#!/usr/bin/env python3
"""
E-Ticketing Revenue Estimator Startup Script
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def main():
    """Main function to run the application"""
    print("🎫 E-Ticketing Revenue Estimator")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ app.py not found. Please run this script from the project directory.")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("🚀 Starting Streamlit application...")
    print("📱 The app will open in your default browser")
    print("🛑 Press Ctrl+C to stop the application")
    print("-" * 40)
    
    try:
        # Check if virtual environment exists
        if os.path.exists("venv/bin/activate"):
            print("🔧 Using virtual environment...")
            # Run the Streamlit app with virtual environment
            subprocess.run([
                "bash", "-c", 
                "source venv/bin/activate && streamlit run app.py"
            ], check=True)
        else:
            print("⚠️ No virtual environment found, running with system Python...")
            # Run the Streamlit app
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 