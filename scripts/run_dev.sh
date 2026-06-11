#!/bin/bash
# AgentFace Development Startup Script
# Starts the main app and both model services in development mode.

set -e

echo "🚀 Starting AgentFace Development Environment..."
echo ""

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Copy .env.example if .env doesn't exist
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "📝 Created .env from .env.example"
fi

# Install dependencies if needed
if [ ! -d "src/agent_face.egg-info" ]; then
    echo "📦 Installing dependencies..."
    pip install -e ".[dev]"
fi

# Start model services in background
echo ""
echo "🔧 Starting Multimodal Model Service (port 8001)..."
cd model_services/multimodal_service
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload &
MULTIMODAL_PID=$!
cd ../..

echo "🔧 Starting Beautification Model Service (port 8002)..."
cd model_services/beautify_service
python -m uvicorn main:app --host 0.0.0.0 --port 8002 --reload &
BEAUTIFY_PID=$!
cd ../..

# Wait for model services to be ready
echo "⏳ Waiting for model services..."
sleep 2

# Start main application
echo ""
echo "🎨 Starting AgentFace Main App (port 8000)..."
python -m uvicorn agent_face.main:app --host 0.0.0.0 --port 8000 --reload &
AGENTFACE_PID=$!

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All services started:"
echo "   AgentFace API:  http://localhost:8000"
echo "   API Docs:       http://localhost:8000/docs"
echo "   Multimodal:     http://localhost:8001"
echo "   Beautification: http://localhost:8002"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to stop all services"

# Handle shutdown
trap "echo '🛑 Stopping all services...'; kill $AGENTFACE_PID $MULTIMODAL_PID $BEAUTIFY_PID 2>/dev/null; exit 0" SIGINT SIGTERM

# Wait for any process to exit
wait
