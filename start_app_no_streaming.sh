#!/bin/bash

# SIGMA Multi-Agent System Startup Script (No Streaming)
# This script starts only the Streamlit app without WebSocket streaming

echo "🚀 Starting SIGMA Multi-Agent System (No Streaming Mode)..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "Please create a .env file with your GOOGLE_API_KEY"
    echo "Example:"
    echo "GOOGLE_API_KEY=your_actual_key_here"
    exit 1
fi

# Source environment variables
export $(cat .env | xargs)

# Check if GOOGLE_API_KEY is set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ Error: GOOGLE_API_KEY not set in .env file"
    exit 1
fi

echo "✅ Environment variables loaded"

# Start Streamlit app
echo "🎨 Starting Streamlit app..."
streamlit run app.py

echo "👋 SIGMA Multi-Agent System stopped"
