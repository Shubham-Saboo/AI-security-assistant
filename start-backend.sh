#!/bin/bash

# Security Assistant Backend Startup Script

echo "🛡️ Starting Security Assistant Backend..."

# Check if .env file exists
if [ ! -f "backend/.env" ]; then
    echo "❌ Error: backend/.env file not found!"
    echo "Please copy backend/env_example.txt to backend/.env and add your OPENAI_API_KEY"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing Python dependencies..."
cd backend
pip install -r requirements.txt

# Check if OpenAI API key is set
if ! grep -q "sk-" .env; then
    echo "⚠️ Warning: Please make sure your OPENAI_API_KEY is set in backend/.env"
fi

# Start the backend server
echo "🚀 Starting FastAPI backend server..."
echo "Backend will be available at: http://localhost:8000"
echo "API documentation at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Load environment and start server
export $(cat .env | xargs) && python -m app.main
