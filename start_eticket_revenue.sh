#!/bin/bash

# E-Ticketing Revenue Estimator Startup Script
# This script activates the virtual environment and starts the Streamlit app

echo "🎫 Starting E-Ticketing Revenue Estimator..."
echo "=============================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run the following commands first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found in current directory!"
    echo "Please make sure you're running this script from the project directory."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found in virtual environment!"
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo "🚀 Starting Streamlit application..."
echo "📱 The app will open in your default browser at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the application"
echo "----------------------------------------------"

# Start the Streamlit app
streamlit run app.py

# Deactivate virtual environment when done
deactivate

echo "👋 E-Ticketing Revenue Estimator stopped." 