#!/bin/bash

# SIGMA Multi-Agent System Startup Script
# This script starts both the WebSocket server and the Streamlit app

echo "🚀 Starting SIGMA Multi-Agent System..."

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

# Start WebSocket server in background
echo "🔌 Starting WebSocket server..."
python websocket_server.py &
WS_PID=$!

# Give the WebSocket server time to start
sleep 2

# Check if WebSocket server started successfully
if ps -p $WS_PID > /dev/null; then
    echo "✅ WebSocket server running (PID: $WS_PID)"
else
    echo "⚠️  WebSocket server failed to start. Continuing without streaming..."
fi

# Start Streamlit app
echo "🎨 Starting Streamlit app..."
streamlit run app.py

# Cleanup on exit
echo ""
echo "🛑 Shutting down..."
if ps -p $WS_PID > /dev/null; then
    kill $WS_PID
    echo "✅ WebSocket server stopped"
fi

echo "👋 SIGMA Multi-Agent System stopped"
