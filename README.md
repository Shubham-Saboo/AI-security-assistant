# 🛡️ AI Security Assistant - Full Stack Evaluation Project

> **Enterprise-grade AI-powered security assistant with agentic capabilities, comprehensive audit logging, and advanced observability.**

## 🎯 Project Overview

This is a complete full-stack application demonstrating advanced AI capabilities in an enterprise security context. The system combines **Generative AI** (LLM-powered conversations) with **Agentic AI** (autonomous tool selection) to create an intelligent security assistant.

### 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────-┐    ┌─────────────────┐
│  React Frontend │    │ FastAPI Backend  │    │ AI & Data Layer │
│                 │    │                  │    │                 │
│ • Chat Interface│◄──►│ • LangGraph Agent│◄──►│ • OpenAI GPT-4o │
│ • Role Selection│    │ • Security Tools │    │ • ChromaDB (RAG)│
│ • Transparency  │    │ • RBAC & Audit   │    │ • Tavily Search │
│ • Web Controls  │    │ • DLP & Masking  │    │ • LangSmith     │
└─────────────────┘    └─────────────────-┘    └─────────────────┘
```

## 🚀 Quick Start for Interviewers

### ⚡ One-Command Setup & Start

```bash
# Clone and setup everything
git clone https://github.com/Shubham-Saboo/AI-security-assistant.git
cd AI-security-assistant
chmod +x start.sh

# Initial setup (first time only)
./start.sh setup

# Start the application
./start.sh
```

**Single Script Commands:**
- `./start.sh setup` - Initial environment setup
- `./start.sh` - Start both backend and frontend  
- `./start.sh backend` - Start backend only
- `./start.sh frontend` - Start frontend only
- `./start.sh status` - Check service status
- `./start.sh stop` - Stop all services
- `./start.sh restart` - Restart everything
- `./start.sh --help` - Show all options

**Access the application at:** http://localhost:3000

### 📋 Manual Setup (if needed)

<details>
<summary>Click to expand manual setup steps</summary>

```bash
# 1. Setup Python Backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
cd backend
pip install -r requirements.txt
cp env_example.txt .env

# 2. Add API Keys to .env
# OPENAI_API_KEY=your_openai_key_here
# LANGSMITH_API_KEY=your_langsmith_key_here (optional)
# TAVILY_API_KEY=your_tavily_key_here (optional)

# 3. Start Backend
python -m app.main

# 4. Setup Frontend (new terminal)
cd ../frontend
npm install
npm start

# 5. Access Application
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

</details>

## 🎯 Core Features Implemented

### 🤖 Agentic AI Capabilities
- **🧠 LangGraph Agent**: Autonomous decision-making and tool selection
- **🔍 Policy Search Tool**: RAG-powered security handbook search  
- **📊 Log Query Tool**: Role-based security log analysis
- **🌐 Web Search Tool**: Real-time threat intelligence via Tavily
- **🎯 Smart Routing**: AI decides which tool to use based on user query

### 🔒 Enterprise Security Features
- **🛡️ Role-Based Access Control (RBAC)**: Security vs Sales role permissions
- **🚫 Prompt Injection Defense**: Detects and blocks malicious inputs
- **🔍 Data Loss Prevention (DLP)**: Automatic masking of sensitive data
- **📝 Comprehensive Audit Logging**: Every action logged with trace correlation
- **🔒 Security Transparency**: "Why did I answer this way?" explanations

### 📊 Advanced Observability  
- **📈 LangSmith Integration**: Full LLM call tracing and performance monitoring
- **🔍 Request Tracing**: Complete visibility into AI decision-making process
- **📊 Performance Metrics**: Token usage, latency, and cost tracking
- **🛠️ Debug Capabilities**: Detailed error analysis and troubleshooting

### 💬 Multi-turn Conversations
- **🧠 Persistent Memory**: LangGraph checkpointer with SQLite storage
- **🔄 Context Retention**: AI remembers conversation history
- **💬 Session Management**: New chat functionality with conversation IDs

## 🧪 Demo Scenarios for Testing

### 👤 Security Role Testing
```bash
# Test policy search
"How should I handle a phishing email?"

# Test log analysis  
"Show me failed login attempts from the last week"

# Test web search
"What are the latest CVE vulnerabilities?"

# Test restricted access
"What is our incident escalation policy?"
```

### 👥 Sales Role Testing  
```bash
# Test basic policy access
"What are our password requirements?"

# Test role-based restrictions
"Show me security logs" (limited view)

# Test web search access
"Find recent cybersecurity news"

# Test document restrictions
"What is our incident escalation policy?" (access denied)
```

### 🔍 Advanced Features Testing
```bash
# Test DLP masking
"Check user john.doe with IP 192.168.1.100"

# Test transparency
Click "🔍 Why did I answer this way?" on any response

# Test conversation memory
Start a conversation, then reference previous messages

# Test web search toggle
Use the toggle in the UI to enable/disable web search
```

## 🏗️ Technical Implementation

### Backend Stack
- **FastAPI**: Modern, fast web framework for APIs
- **LangChain + LangGraph**: Agentic AI framework with graph-based workflows  
- **OpenAI GPT-4o-mini**: Large language model for conversations
- **ChromaDB**: Vector database for document retrieval (RAG)
- **LangSmith**: LLM observability and performance monitoring
- **Tavily**: Real-time web search capabilities

### Frontend Stack  
- **React + TypeScript**: Modern frontend with type safety
- **CSS3**: Custom styling with responsive design
- **Fetch API**: HTTP client for backend communication

### Data & Security
- **Role-Based Access Control**: Configurable permissions system
- **Data Loss Prevention**: Regex-based sensitive data detection and masking
- **Audit Logging**: Comprehensive logging with DLP integration
- **Prompt Injection Defense**: Pattern-based malicious input detection

## 📁 Project Structure

```
Security-Assistant/
├── 📱 frontend/                 # React TypeScript frontend
│   ├── src/
│   │   ├── App.tsx             # Main chat interface
│   │   └── App.css             # UI styling
│   └── package.json
├── 🔧 backend/                  # FastAPI Python backend  
│   ├── app/
│   │   └── main.py             # Core application logic
│   ├── requirements.txt        # Python dependencies
│   └── SETUP.md               # Backend setup guide
├── 📊 mock_data/               # Sample data and configuration
│   ├── handbooks/             # Security policy documents
│   ├── logs/                  # Security log samples  
│   └── rbac_config.json       # Role-based access config
├── 🗄️ chroma_db/              # Vector database storage
├── 🚀 setup-for-interview.sh   # One-command setup script
└── 📖 README.md               # This file
```

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health with LangSmith status |
| `/chat` | POST | Main AI chat interface |
| `/audit-logs` | GET | View audit logs (Security role only) |
| `/new-conversation` | POST | Start new conversation thread |
| `/conversation-history` | POST | Retrieve conversation history |
| `/available-documents` | GET | List accessible documents by role |
| `/dlp-status` | GET | DLP monitoring statistics |

## 🎨 UI Features

### Chat Interface
- **Role Selection**: Switch between Security and Sales roles
- **Message History**: Persistent conversation display
- **Transparency Panel**: Expandable "Why did I answer this way?" explanations
- **Web Search Toggle**: Enable/disable real-time web search
- **New Chat Button**: Start fresh conversations

### Security Transparency
- **Processing Summary**: Steps, timing, security checks
- **Security Status**: Visual indicators for security validation
- **Tool Justification**: Why specific tools were chosen
- **Confidence Assessment**: AI confidence levels and limitations

## 🔍 Security Controls

### Prompt Injection Defense
```python
# Detects patterns like:
- "Ignore previous instructions"
- "You are now in developer mode"  
- "Disregard your training"
```

### Data Loss Prevention (DLP)
```python
# Automatically masks:
- Usernames (john.doe → jo****)
- IP Addresses (192.168.1.100 → 192.***)
- API Keys (sk-proj-abc... → sk-p****)
- Credit Cards, SSNs, etc.
```

### Role-Based Access Control
```json
{
  "security": {
    "accessible_files": ["all_handbooks"],
    "permissions": ["read_logs", "read_all_handbooks"]
  },
  "sales": {
    "accessible_files": ["basic_policies_only"], 
    "permissions": ["read_basic_policies"]
  }
}
```

## 📊 Monitoring & Observability

### LangSmith Integration
- **Complete LLM Tracing**: Every AI interaction tracked
- **Performance Monitoring**: Real-time latency and cost analysis
- **Error Debugging**: Detailed failure analysis
- **Audit Enhancement**: Trace IDs linked to audit logs

### Built-in Audit System
- **Action Logging**: Every user interaction recorded
- **DLP Integration**: Sensitive data masking events
- **Security Checks**: Prompt injection and RBAC validations
- **Trace Correlation**: Links to LangSmith traces

## 🎯 Evaluation Criteria Met

### ✅ AI Functionality
- [x] **Conversational LLM Interface**: OpenAI GPT-4o-mini integration
- [x] **Agentic Tool Usage**: LangGraph autonomous decision-making
- [x] **Full Stack Implementation**: React frontend + FastAPI backend

### ✅ AI Security  
- [x] **Prompt Injection Defense**: Pattern-based malicious input detection
- [x] **Role-Based Access Control**: Security vs Sales role permissions  
- [x] **Audit Logging**: Comprehensive action tracking with trace correlation

### ✅ Stretch Goals (All Implemented!)
- [x] **Multiple Tools**: Policy search + Log query + Web search
- [x] **Conversation Memory**: Multi-turn persistent conversations
- [x] **RAG Implementation**: ChromaDB vector search with embeddings
- [x] **Data Masking (DLP)**: Automatic sensitive data protection
- [x] **Security Transparency**: Complete decision explanations

## 🚀 Production Deployment

The application is ready for production deployment with:
- **Docker containerization** (configurable)
- **Cloud platform deployment** (Railway, Render, AWS, etc.)
- **Environment-based configuration**
- **Comprehensive monitoring and alerting**

## 📝 License

This project is created for evaluation purposes and demonstrates enterprise-grade AI security assistant capabilities.

---

**🎯 Ready for Interview Demo!** 

Start with: `./setup-for-interview.sh` and access http://localhost:3000

For questions or issues, the complete setup documentation is in `backend/SETUP.md`