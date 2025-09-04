# Security Assistant Backend Setup

## Quick Start

1. **Create .env file** (copy from env_example.txt):
   ```bash
   cp env_example.txt .env
   ```

2. **Add your API keys** to the .env file:
   ```
   OPENAI_API_KEY=sk-your-actual-openai-api-key-here
   TAVILY_API_KEY=tvly-your-tavily-api-key-here  # Optional for web search
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
   - `/health`: Health check
   - `/audit-logs`: View audit logs (security role only)
   - `/available-documents`: List accessible documents by role

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
1. Decide which tool to use (policy search vs log query)
2. Filter results based on user role
3. Provide relevant, role-appropriate responses
4. Log all actions for audit purposes
