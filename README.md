# 🛡️ Security Assistant - AI Evaluation Task

An enterprise-grade AI-powered security assistant with agentic capabilities, built for the Aegeos AI Full Stack Engineer evaluation.

## 📋 Project Overview

This application demonstrates a working prototype of a GenAI + Agentic AI full stack application in the context of enterprise security, featuring:

- **Conversational AI** powered by OpenAI GPT-4o-mini
- **Agentic behavior** with autonomous tool selection using LangGraph
- **Retrieval-Augmented Generation (RAG)** using ChromaDB and OpenAI embeddings
- **Role-Based Access Control (RBAC)** for Security vs Sales teams
- **Security controls** including prompt injection defense and audit logging
- **Full stack implementation** with React frontend and FastAPI backend

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- Node.js 16+
- OpenAI API Key

### Option 1: Start Everything (Recommended)
```bash
# 1. Set up your OpenAI API key
cp backend/env_example.txt backend/.env
# Edit backend/.env and add your OPENAI_API_KEY

# 2. Start both backend and frontend
./start-all.sh
```

### Option 2: Start Services Separately
```bash
# Terminal 1 - Backend
./start-backend.sh

# Terminal 2 - Frontend  
./start-frontend.sh
```

### Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🏗️ Architecture

### Backend (FastAPI + LangChain + ChromaDB)
```
├── app/main.py              # Complete backend (623 lines)
├── requirements.txt         # Python dependencies
└── SETUP.md                # Backend setup guide
```

### Frontend (React TypeScript)
```
├── src/App.tsx              # Main chat interface
├── src/App.css              # Complete styling
└── README.md               # Frontend guide
```

### Data & Configuration
```
├── mock_data/
│   ├── handbooks/          # 3 security policy documents
│   ├── logs/              # CSV security logs
│   └── rbac_config.json   # Role permissions
└── chroma_db/             # Vector embeddings (auto-generated)
```

## 🎯 Core Features

### ✅ Generative AI Chat Interface
- React TypeScript frontend with real-time chat
- OpenAI GPT-4o-mini for natural language processing
- Conversation memory and context management

### ✅ Agentic Tool Integration
- **Policy Search Tool**: RAG-based search of security handbooks
- **Log Query Tool**: CSV log analysis with filtering
- **Autonomous Decision Making**: AI decides which tool to use

### ✅ Role-Based Access Control (RBAC)
- **Security Team**: Full access to all policies and logs
- **Sales Team**: Limited access (no incident escalation procedures)
- Document-level permissions with metadata filtering

### ✅ AI Security Controls
- **Prompt Injection Defense**: Detects and blocks malicious inputs
- **Audit Logging**: Comprehensive logging of all actions
- **Data Masking**: Role-based information filtering

## 🧪 Evaluation Scenarios

### Test Case 1: Phishing Email Handler
```
User: "How should I handle a suspected phishing email?"
Assistant: Retrieves phishing policy → summarizes in plain English
```

### Test Case 2: Log Analysis
```
User: "Show me today's failed login attempts from the logs"
Assistant: Queries CSV logs → summarizes results with context
```

### Test Case 3: Incident Escalation (Security Role Only)
```
User: "What's the escalation path for a security breach?"
Assistant: Retrieves playbook steps → provides detailed procedures
```

## 🔐 Security Features

### Document Access Control
- **Public**: `phishing_response.md`, `general_security_policy.md`
- **Restricted**: `incident_escalation.md` (Security team only)

### User Roles
- **Security Team**: Full access + critical incident procedures
- **Sales Team**: Basic policies + filtered logs (sales dept only)

### Audit Trail
All interactions logged with:
- Timestamp and user role
- Query content and tool usage
- Response summaries
- Security events (prompt injection attempts)

## 📊 Technical Implementation

### RAG Pipeline
1. **Document Processing**: RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
2. **Embeddings**: OpenAI text-embedding-3-small
3. **Vector Storage**: ChromaDB with role-based metadata
4. **Retrieval**: Similarity search with role filtering

### Agent Architecture
```python
# LangGraph ReAct Agent
tools = [PolicySearchTool(), LogQueryTool()]
agent = create_react_agent(llm, tools, checkpointer=memory)
```

### API Endpoints
- `POST /chat` - Main chat interface
- `GET /health` - System health check
- `GET /audit-logs` - Audit trail (Security only)
- `GET /available-documents` - User's accessible files

## 🛠️ Development

### Project Structure
```
Security-Assistant/
├── backend/                 # FastAPI backend
├── frontend/               # React frontend  
├── mock_data/              # Sample data
├── start-*.sh             # Startup scripts
├── .gitignore             # Version control
└── README.md              # This file
```

### Key Technologies
- **Backend**: FastAPI, LangChain, ChromaDB, OpenAI
- **Frontend**: React 19, TypeScript, CSS3
- **AI**: OpenAI GPT-4o-mini, LangGraph, RAG
- **Security**: RBAC, Input validation, Audit logging

## 📝 Deliverables

### ✅ Source Code
- Complete working prototype
- Clean, documented codebase
- Easy-to-run startup scripts

### ✅ Documentation
- Setup instructions (this README)
- Architecture notes in backend/SETUP.md
- Frontend guide in frontend/README.md

### ✅ Working Demo
- Full-stack application ready for demonstration
- Multiple test scenarios implemented
- Security controls demonstrated

## 🎬 Demo Script

1. **Role Selection**: Choose Security vs Sales role
2. **Policy Query**: "How should I handle a phishing email?"
3. **Log Analysis**: "Show me failed login attempts"
4. **RBAC Demo**: Try accessing incident escalation as Sales user
5. **Security Test**: Attempt prompt injection
6. **Audit Review**: View comprehensive audit logs

## 🚀 Production Deployment

For production deployment:
1. Set proper environment variables
2. Use production OpenAI API limits
3. Implement proper authentication
4. Add HTTPS/TLS encryption
5. Scale ChromaDB for production use

## 📞 Support

For questions about this evaluation project:
- Check the backend/SETUP.md for detailed setup
- View frontend/README.md for UI specifics
- Review audit logs for troubleshooting

---

**Built for Aegeos AI Full Stack Engineer Evaluation**  
*Demonstrating enterprise-grade AI security assistant capabilities*
