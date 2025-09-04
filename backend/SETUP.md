# Security Assistant Backend Setup

## Quick Start

1. **Create .env file** (copy from env_example.txt):
   ```bash
   cp env_example.txt .env
   ```

2. **Add your API keys** to the .env file:
   ```
   OPENAI_API_KEY=sk-your-actual-openai-api-key-here
   TAVILY_API_KEY=tvly-your-tavily-api-key-here         # Optional for web search
   LANGSMITH_API_KEY=lsv2_your-langsmith-api-key-here   # Optional for observability
   LANGSMITH_TRACING=true                               # Enable LangSmith tracing
   LANGSMITH_PROJECT=security-assistant                 # LangSmith project name
   ```

3. **Run the backend**:
   ```bash
   source ../venv/bin/activate
   python -m app.main
   ```

4. **Test the API**:
   - Open http://localhost:8000 in your browser
   - API documentation: http://localhost:8000/docs

## What's Included

### ✅ Features Implemented:

1. **Document Processing & ChromaDB**:
   - Automatic chunking of security handbooks
   - Role-based metadata for access control
   - Vector embeddings using OpenAI's text-embedding-3-small

2. **Agentic AI Tools**:
   - `PolicySearchTool`: Search security policies with RAG
   - `LogQueryTool`: Query security logs with role-based filtering
   - `ThreatIntelligenceTool`: Real-time web search for threat intelligence using Tavily
   - LangGraph agent that autonomously decides which tool to use

3. **Security Controls**:
   - **Prompt Injection Defense**: Detects malicious input patterns
   - **RBAC**: Role-based access (Security vs Sales roles)
   - **Audit Logging**: All actions logged with timestamp, user, and results

4. **API Endpoints**:
   - `/chat`: Main chat interface with AI assistant
   - `/health`: Health check with LangSmith integration status
   - `/audit-logs`: View audit logs with LangSmith trace IDs (security role only)
   - `/available-documents`: List accessible documents by role

5. **Enhanced Observability with LangSmith** (Optional):
   - **Full LLM Call Tracing**: Complete visibility into every AI interaction
   - **Performance Monitoring**: Latency, token usage, and cost tracking
   - **Error Debugging**: Detailed trace analysis for troubleshooting
   - **Audit Integration**: Trace IDs linked to audit logs for complete accountability
   - **Production Monitoring**: Real-time dashboards and alerting

### Role-Based Access:

**Security Role**:
- Access to all 3 handbooks (including incident_escalation.md)
- Can view all security logs
- Can query critical/high-risk events

**Sales Role**:
- Access to 2 handbooks (phishing_response.md, general_security_policy.md) 
- Can only see sales/marketing department logs
- Limited to low/medium risk events
- Cannot access incident_escalation.md

## LangSmith Setup (Optional - For Enhanced Observability)

To enable LangSmith tracing for production-grade observability:

1. **Sign up for LangSmith**: Visit https://smith.langchain.com/
2. **Create an API key**: Go to Settings → API Keys → Create API Key
3. **Add to .env file**:
   ```
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=lsv2_your-actual-langsmith-api-key
   LANGSMITH_PROJECT=security-assistant
   ```
4. **Restart the backend** to enable tracing

### LangSmith Benefits:
- **Complete Trace Visibility**: See every step of LLM processing
- **Performance Analytics**: Monitor latency, costs, and token usage  
- **Error Debugging**: Detailed analysis of failures and edge cases
- **Audit Enhancement**: Trace IDs automatically linked to audit logs
- **Production Monitoring**: Real-time dashboards and automated alerts

### Without LangSmith:
The application works perfectly without LangSmith - you'll still have:
- ✅ Basic audit logging with DLP masking
- ✅ Security transparency explanations
- ✅ Role-based access control
- ✅ All core functionality

## Test Your Setup

Run the test script:
```bash
python test_setup.py
```

## Example API Usage

### Chat with Security Role:
```bash
curl -X POST "http://localhost:8000/chat" \
-H "Content-Type: application/json" \
-d '{
  "message": "How should I handle a phishing email?",
  "user_role": "security"
}'
```

### Chat with Sales Role:
```bash
curl -X POST "http://localhost:8000/chat" \
-H "Content-Type: application/json" \
-d '{
  "message": "Show me recent login failures",
  "user_role": "sales"
}'
```

The AI assistant will automatically:
1. Decide which tool to use (policy search, log query, or web search)
2. Filter results based on user role and RBAC permissions
3. Apply DLP masking to sensitive data in responses
4. Provide comprehensive transparency explanations
5. Maintain conversation memory across multiple turns
6. Log all actions with LangSmith trace correlation for audit purposes

## 🎯 Complete Feature Overview

### ✅ Implemented Features:

#### 🤖 Agentic AI Core:
- **LangGraph Agent**: Autonomous decision-making with ReAct pattern
- **Multi-tool Routing**: Intelligent selection between 3 specialized tools
- **Conversation Memory**: Persistent multi-turn conversations with SQLite checkpointer
- **Dynamic Tool Selection**: Web search can be toggled on/off per conversation

#### 🔍 AI Tools:
1. **PolicySearchTool**: RAG-powered security handbook search with ChromaDB
2. **LogQueryTool**: Role-based security log analysis with CSV data
3. **WebSearchTool**: Real-time threat intelligence via Tavily API

#### 🔒 Enterprise Security:
- **Role-Based Access Control (RBAC)**: Security vs Sales with granular permissions
- **Prompt Injection Defense**: Pattern-based malicious input detection
- **Data Loss Prevention (DLP)**: 9 sensitive data patterns with role-based masking
- **Comprehensive Audit Logging**: Every action logged with DLP integration
- **Security Transparency**: Complete AI decision explanations

#### 📊 Advanced Observability:
- **LangSmith Integration**: Full LLM tracing and performance monitoring
- **Request Tracing**: Complete visibility into agent decision-making
- **Performance Metrics**: Token usage, latency, and cost tracking
- **Error Debugging**: Detailed trace analysis for troubleshooting

## 🎬 Demo & Testing

For comprehensive demo scenarios and testing instructions, see:
- **DEMO_GUIDE.md**: Complete testing scenarios for all features
- **Root README.md**: Project overview and quick start guide
