#!/bin/bash

# 🛡️ AI Security Assistant - Universal Startup Script
# Handles setup, backend-only, frontend-only, or full application startup

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${CYAN}$1${NC}"
}

# Function to show usage
show_usage() {
    echo ""
    print_header "🛡️  AI Security Assistant - Universal Startup Script"
    echo "============================================================"
    echo ""
    echo "Usage: ./start.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "COMMANDS:"
    echo "  setup         Complete initial setup (install dependencies, configure)"
    echo "  backend       Start backend API server only"
    echo "  frontend      Start frontend React app only"
    echo "  all           Start both backend and frontend (default)"
    echo "  status        Check if services are running"
    echo "  stop          Stop all running services"
    echo "  restart       Restart all services"
    echo "  clean         Clean temporary files and restart"
    echo ""
    echo "OPTIONS:"
    echo "  --help, -h    Show this help message"
    echo "  --verbose, -v Verbose output"
    echo ""
    echo "EXAMPLES:"
    echo "  ./start.sh                    # Start both services"
    echo "  ./start.sh setup              # Initial setup for new environment"
    echo "  ./start.sh backend            # Start only backend API"
    echo "  ./start.sh frontend           # Start only frontend UI"
    echo "  ./start.sh status             # Check service status"
    echo "  ./start.sh restart            # Restart all services"
    echo ""
    echo "ACCESS POINTS:"
    echo "  Frontend UI:        http://localhost:3000"
    echo "  Backend API:        http://localhost:8000"
    echo "  API Documentation:  http://localhost:8000/docs"
    echo ""
}

# Function to check if we're in the right directory
check_directory() {
    if [ ! -f "README.md" ] || [ ! -d "backend" ] || [ ! -d "frontend" ]; then
        print_error "Please run this script from the Security-Assistant root directory"
        exit 1
    fi
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is required but not installed"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to setup environment
setup_environment() {
    print_header "🔧 Setting up AI Security Assistant..."
    echo "========================================"
    echo ""
    
    check_prerequisites
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    fi
    
    # Install Python dependencies
    print_status "Installing Python dependencies..."
    source venv/bin/activate
    cd backend
    pip install --upgrade pip
    pip install -r requirements.txt
    cd ..
    print_success "Python dependencies installed"
    
    # Setup environment file
    if [ ! -f "backend/.env" ]; then
        print_status "Creating environment configuration..."
        cd backend
        cp env_example.txt .env
        cd ..
        print_success "Environment file created from template"
        print_warning "IMPORTANT: Add your API keys to backend/.env"
        print_warning "Required: OPENAI_API_KEY"
        print_warning "Optional: LANGSMITH_API_KEY, TAVILY_API_KEY"
    fi
    
    # Install Node.js dependencies
    print_status "Installing Node.js dependencies..."
    cd frontend
    if [ ! -d "node_modules" ]; then
        npm install
        print_success "Node.js dependencies installed"
    else
        print_success "Node.js dependencies already installed"
    fi
    cd ..
    
    print_success "Setup completed successfully!"
    echo ""
    print_warning "Next steps:"
    echo "1. Edit backend/.env with your API keys"
    echo "2. Run: ./start.sh all"
    echo ""
}

# Function to check service status
check_status() {
    print_header "📊 Service Status Check"
    echo "========================"
    echo ""
    
    # Check backend
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_success "Backend API: Running (http://localhost:8000)"
    else
        print_warning "Backend API: Not running"
    fi
    
    # Check frontend
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        print_success "Frontend UI: Running (http://localhost:3000)"
    else
        print_warning "Frontend UI: Not running"
    fi
    echo ""
}

# Function to stop services
stop_services() {
    print_header "🛑 Stopping Services"
    echo "====================="
    echo ""
    
    print_status "Stopping backend processes..."
    pkill -f "python.*main" 2>/dev/null || true
    
    print_status "Stopping frontend processes..."
    pkill -f "npm start" 2>/dev/null || true
    pkill -f "react-scripts start" 2>/dev/null || true
    
    # Wait a moment for processes to stop
    sleep 2
    
    print_success "All services stopped"
}

# Function to clean temporary files
clean_temp_files() {
    print_status "Cleaning temporary files..."
    
    # Remove conversation database files
    rm -f backend/conversations.db* 2>/dev/null || true
    
    # Remove log files
    rm -f backend/server.log backend/*.log *.log 2>/dev/null || true
    
    # Remove Python cache
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    print_success "Temporary files cleaned"
}

# Function to start backend
start_backend() {
    print_header "🚀 Starting Backend API Server"
    echo "==============================="
    echo ""
    
    # Check if backend is already running
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_warning "Backend already running at http://localhost:8000"
        return
    fi
    
    print_status "Activating Python virtual environment..."
    source venv/bin/activate
    
    print_status "Starting FastAPI backend server..."
    cd backend
    
    # Check if .env exists
    if [ ! -f ".env" ]; then
        print_error "Backend .env file not found. Run: ./start.sh setup"
        exit 1
    fi
    
    # Start backend in background
    if [ "$VERBOSE" = "true" ]; then
        python -m app.main &
    else
        python -m app.main > ../backend.log 2>&1 &
    fi
    
    BACKEND_PID=$!
    cd ..
    
    # Wait for backend to start
    print_status "Waiting for backend to start..."
    for i in {1..15}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            print_success "Backend started successfully!"
            print_success "API available at: http://localhost:8000"
            print_success "Documentation at: http://localhost:8000/docs"
            return
        fi
        sleep 1
    done
    
    print_error "Backend failed to start within 15 seconds"
    if [ "$VERBOSE" != "true" ]; then
        echo "Check backend.log for errors"
    fi
    exit 1
}

# Function to start frontend
start_frontend() {
    print_header "🌐 Starting Frontend React App"
    echo "==============================="
    echo ""
    
    # Check if frontend is already running
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        print_warning "Frontend already running at http://localhost:3000"
        return
    fi
    
    print_status "Starting React development server..."
    cd frontend
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        print_error "Node modules not found. Run: ./start.sh setup"
        exit 1
    fi
    
    # Start frontend in background
    if [ "$VERBOSE" = "true" ]; then
        npm start &
    else
        npm start > ../frontend.log 2>&1 &
    fi
    
    FRONTEND_PID=$!
    cd ..
    
    # Wait for frontend to start
    print_status "Waiting for frontend to start..."
    for i in {1..30}; do
        if curl -s http://localhost:3000 > /dev/null 2>&1; then
            print_success "Frontend started successfully!"
            print_success "UI available at: http://localhost:3000"
            return
        fi
        sleep 1
    done
    
    print_error "Frontend failed to start within 30 seconds"
    if [ "$VERBOSE" != "true" ]; then
        echo "Check frontend.log for errors"
    fi
    exit 1
}

# Function to start all services
start_all() {
    print_header "🚀 Starting AI Security Assistant (Full Stack)"
    echo "==============================================="
    echo ""
    
    # Function to cleanup background processes on exit
    cleanup() {
        echo ""
        print_status "Shutting down services..."
        if [ ! -z "$BACKEND_PID" ]; then
            kill $BACKEND_PID 2>/dev/null || true
        fi
        if [ ! -z "$FRONTEND_PID" ]; then
            kill $FRONTEND_PID 2>/dev/null || true
        fi
        pkill -f "python.*main" 2>/dev/null || true
        pkill -f "npm start" 2>/dev/null || true
        print_success "Services stopped"
        exit 0
    }
    
    # Set up signal handlers for graceful shutdown
    trap cleanup SIGINT SIGTERM
    
    # Start backend
    start_backend
    
    # Start frontend
    start_frontend
    
    echo ""
    print_success "🎉 AI Security Assistant is running!"
    echo "===================================="
    echo ""
    echo "🌐 Frontend UI:       http://localhost:3000"
    echo "📡 Backend API:       http://localhost:8000"
    echo "📚 API Docs:          http://localhost:8000/docs"
    echo ""
    echo "📋 Demo Scenarios:"
    echo "   - Try: 'How should I handle a phishing email?'"
    echo "   - Try: 'Show me recent security logs'"
    echo "   - Enable web search for: 'Latest CVE vulnerabilities'"
    echo ""
    echo "Press Ctrl+C to stop all services"
    echo ""
    
    # Wait for user to stop services
    if [ "$BACKGROUND" != "true" ]; then
        while true; do
            sleep 1
        done
    fi
}

# Parse command line arguments
COMMAND="all"
VERBOSE="false"
BACKGROUND="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        setup)
            COMMAND="setup"
            shift
            ;;
        backend)
            COMMAND="backend"
            shift
            ;;
        frontend)
            COMMAND="frontend"
            shift
            ;;
        all)
            COMMAND="all"
            shift
            ;;
        status)
            COMMAND="status"
            shift
            ;;
        stop)
            COMMAND="stop"
            shift
            ;;
        restart)
            COMMAND="restart"
            shift
            ;;
        clean)
            COMMAND="clean"
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        --verbose|-v)
            VERBOSE="true"
            shift
            ;;
        --background|-b)
            BACKGROUND="true"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check directory
check_directory

# Execute command
case $COMMAND in
    setup)
        setup_environment
        ;;
    backend)
        start_backend
        if [ "$BACKGROUND" != "true" ]; then
            echo "Press Ctrl+C to stop the backend"
            wait
        fi
        ;;
    frontend)
        start_frontend
        if [ "$BACKGROUND" != "true" ]; then
            echo "Press Ctrl+C to stop the frontend"
            wait
        fi
        ;;
    all)
        start_all
        ;;
    status)
        check_status
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        sleep 2
        start_all
        ;;
    clean)
        stop_services
        clean_temp_files
        start_all
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac
