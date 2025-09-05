# 🛡️ AI Security Assistant

> **Enterprise-grade AI-powered security assistant with agentic capabilities, comprehensive audit logging, and advanced observability.**

## 🎯 Overview

An intelligent security assistant that combines **Generative AI** (LLM-powered conversations) with **Agentic AI** (autonomous tool selection) to help security teams with incident response, policy guidance, and threat intelligence.

### ✨ Key Features

- **🤖 Agentic AI**: Autonomous decision-making and tool selection using LangGraph
- **🔍 Smart Tools**: Policy search, log analysis, and real-time web search
- **🛡️ Enterprise Security**: Role-based access control, DLP, prompt injection defense
- **📊 Observability**: LangSmith integration with full tracing and transparency
- **💬 Memory**: Multi-turn conversations with persistent context
- **🎨 Rich UI**: Markdown rendering with security transparency explanations

## 🚀 Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/Shubham-Saboo/AI-security-assistant.git
cd AI-security-assistant

# 2. Setup and start
./start.sh setup && ./start.sh
```

**Access the application:** http://localhost:3000

### 📋 Available Commands

```bash
./start.sh setup      # Initial environment setup
./start.sh            # Start both backend and frontend
./start.sh backend    # Start backend only
./start.sh frontend   # Start frontend only  
./start.sh status     # Check service status
./start.sh stop       # Stop all services
./start.sh restart    # Restart everything
./start.sh --help     # Show all options
```

## 🔑 Configuration

Add your API keys to `backend/.env`:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (for enhanced features)
LANGSMITH_API_KEY=your_langsmith_api_key_here    # Enhanced observability
TAVILY_API_KEY=your_tavily_api_key_here          # Web search capabilities
```

## 🧪 Demo Scenarios

### Security Role
- "How should I handle a phishing email?"
- "Show me recent failed login attempts"
- "What are the latest CVE vulnerabilities?" (requires web search)

### Sales Role  
- "What are our password requirements?"
- "Show me security logs" (limited access)

## 🔒 Security Features

- **Role-Based Access Control**: Security vs Sales permissions
- **Data Loss Prevention**: Automatic masking of sensitive data
- **Prompt Injection Defense**: Malicious input detection
- **Audit Logging**: Complete action tracking with trace correlation
- **Security Transparency**: "Why did I answer this way?" explanations

## 📊 Architecture

Built with:
- **Backend**: FastAPI + LangChain + LangGraph
- **Frontend**: React + TypeScript
- **AI**: OpenAI GPT-4o-mini with agentic capabilities
- **Data**: ChromaDB (vector search) + CSV logs + Markdown policies
- **Observability**: LangSmith integration

For detailed architecture and design notes, see [ARCHITECTURE.md](ARCHITECTURE.md).

## 📚 Documentation

- **[DEMO_GUIDE.md](DEMO_GUIDE.md)**: Complete testing scenarios and examples
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Technical design and implementation details

## 🎯 Project Status

✅ **Production Ready**: Enterprise-grade security assistant with comprehensive features  
✅ **Interview Ready**: Simple setup with professional demonstration scenarios  
✅ **Well Documented**: Complete guides for setup, testing, and architecture

---

**Ready for demonstration and evaluation!** 🚀