#!/bin/bash

# 🛡️ AI Security Assistant - Interview Setup Script
# One-command setup for evaluators/interviewers

set -e  # Exit on any error

echo "🛡️  AI Security Assistant - Interview Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    print_error "Please run this script from the Security-Assistant root directory"
    exit 1
fi

print_status "Starting AI Security Assistant setup..."
echo ""

# Step 1: Check Python version
print_status "1/8 Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python found: $PYTHON_VERSION"
else
    print_error "Python 3 is required but not installed"
    echo "Please install Python 3.8 or higher and try again"
    exit 1
fi

# Step 2: Check Node.js version  
print_status "2/8 Checking Node.js version..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_success "Node.js found: $NODE_VERSION"
else
    print_error "Node.js is required but not installed"
    echo "Please install Node.js 16+ and try again"
    exit 1
fi

# Step 3: Create Python virtual environment
print_status "3/8 Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Step 4: Activate virtual environment and install Python dependencies
print_status "4/8 Installing Python dependencies..."
source venv/bin/activate
cd backend
pip install --upgrade pip
pip install -r requirements.txt
print_success "Python dependencies installed"

# Step 5: Setup environment file
print_status "5/8 Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp env_example.txt .env
    print_success "Environment file created from template"
    print_warning "IMPORTANT: You need to add your API keys to backend/.env"
    print_warning "Required: OPENAI_API_KEY"
    print_warning "Optional: LANGSMITH_API_KEY, TAVILY_API_KEY"
else
    print_success "Environment file already exists"
fi

# Check if OpenAI key is configured
if grep -q "your_openai_api_key_here" .env; then
    print_warning "⚠️  OpenAI API key not configured in backend/.env"
    print_warning "The application will not work without an OpenAI API key"
    echo ""
    echo "To configure:"
    echo "1. Edit backend/.env"  
    echo "2. Replace 'your_openai_api_key_here' with your actual OpenAI API key"
    echo "3. Optionally add LANGSMITH_API_KEY and TAVILY_API_KEY"
    echo ""
    read -p "Press Enter to continue setup (you can add keys later)..."
fi

cd ..

# Step 6: Install Node.js dependencies
print_status "6/8 Installing Node.js dependencies..."
cd frontend
if [ ! -d "node_modules" ]; then
    npm install
    print_success "Node.js dependencies installed"
else
    print_success "Node.js dependencies already installed"
fi
cd ..

# Step 7: Test backend startup
print_status "7/8 Testing backend startup..."
cd backend
source ../venv/bin/activate

# Check if backend starts successfully
timeout 10 python -m app.main &
BACKEND_PID=$!
sleep 5

if kill -0 $BACKEND_PID 2>/dev/null; then
    print_success "Backend started successfully"
    kill $BACKEND_PID
    wait $BACKEND_PID 2>/dev/null || true
else
    print_warning "Backend startup test failed (likely due to missing API keys)"
fi

cd ..

# Step 8: Create startup scripts
print_status "8/8 Creating startup scripts..."

# Backend startup script
cat > start-backend.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting AI Security Assistant Backend..."
cd backend
source ../venv/bin/activate
python -m app.main
EOF

# Frontend startup script  
cat > start-frontend.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting AI Security Assistant Frontend..."
cd frontend
npm start
EOF

# Combined startup script
cat > start-all.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting AI Security Assistant (Backend + Frontend)..."

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    exit 0
}

# Set up signal handlers for graceful shutdown
trap cleanup SIGINT SIGTERM

echo "📡 Starting backend server..."
cd backend
source ../venv/bin/activate
python -m app.main &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 5

echo "🌐 Starting frontend server..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Both services started!"
echo "🌐 Frontend: http://localhost:3000"
echo "📡 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for both processes
wait $BACKEND_PID
wait $FRONTEND_PID
EOF

# Make scripts executable
chmod +x start-backend.sh start-frontend.sh start-all.sh

print_success "Startup scripts created"

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📋 Next Steps:"
echo ""
if grep -q "your_openai_api_key_here" backend/.env; then
    echo "⚠️  1. Configure API keys in backend/.env:"
    echo "   - OPENAI_API_KEY (required)"
    echo "   - LANGSMITH_API_KEY (optional for observability)"  
    echo "   - TAVILY_API_KEY (optional for web search)"
    echo ""
fi

echo "🚀 2. Start the application:"
echo "   ./start.sh                        # Start both backend and frontend"
echo "   ./start.sh backend                # Start backend only"  
echo "   ./start.sh frontend               # Start frontend only"
echo "   ./start.sh status                 # Check service status"
echo ""

echo "🌐 3. Access the application:"
echo "   Frontend UI: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Documentation: http://localhost:8000/docs"
echo ""

echo "🧪 4. Test the application:"
echo "   - Select Security or Sales role"
echo "   - Try: 'How should I handle a phishing email?'"
echo "   - Try: 'Show me recent security logs'"
echo "   - Enable web search and try: 'Latest CVE vulnerabilities'"
echo ""

if grep -q "your_openai_api_key_here" backend/.env; then
    print_warning "Remember: The application requires an OpenAI API key to function!"
    print_warning "Edit backend/.env and replace 'your_openai_api_key_here'"
else
    print_success "Ready to run! Execute: ./start-all.sh"
fi

echo ""
echo "📖 For detailed documentation, see README.md"
echo "🔧 For technical setup details, see backend/SETUP.md"
echo ""
echo "Happy testing! 🛡️"
