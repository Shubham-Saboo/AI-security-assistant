#!/bin/bash

# Security Assistant - Start Both Backend and Frontend

echo "🛡️ Security Assistant - Full Stack Startup"
echo "=========================================="

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    if [[ ! -z "$BACKEND_PID" ]]; then
        kill $BACKEND_PID 2>/dev/null
        echo "Backend stopped"
    fi
    if [[ ! -z "$FRONTEND_PID" ]]; then
        kill $FRONTEND_PID 2>/dev/null
        echo "Frontend stopped"
    fi
    exit 0
}

# Trap cleanup function on script exit
trap cleanup SIGINT SIGTERM EXIT

# Check if .env file exists
if [ ! -f "backend/.env" ]; then
    echo "❌ Error: backend/.env file not found!"
    echo "Please copy backend/env_example.txt to backend/.env and add your OPENAI_API_KEY"
    exit 1
fi

# Start backend in background
echo "🔧 Starting backend..."
./start-backend.sh > backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Backend failed to start. Check backend.log for details."
        exit 1
    fi
    sleep 2
done

# Start frontend in background
echo "🔧 Starting frontend..."
cd frontend
npm start > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
echo "⏳ Waiting for frontend to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "✅ Frontend is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Frontend failed to start. Check frontend.log for details."
        exit 1
    fi
    sleep 2
done

echo ""
echo "🎉 Security Assistant is now running!"
echo "=================================="
echo "Frontend: http://localhost:3000"
echo "Backend:  http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Logs:"
echo "- Backend: backend.log"
echo "- Frontend: frontend.log"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Keep script running and show live logs
tail -f backend.log frontend.log
