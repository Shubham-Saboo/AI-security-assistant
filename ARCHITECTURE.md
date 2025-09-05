# 🏗️ AI Security Assistant - Architecture & Design

> **Technical design and implementation details for the AI-powered security assistant**

## 🎯 System Overview

The AI Security Assistant is a full-stack application that demonstrates enterprise-grade AI capabilities in a security context, combining **Generative AI** (conversational interface) with **Agentic AI** (autonomous tool selection) to create an intelligent security assistant.

### 🏛️ High-Level Architecture

```
┌─────────────────┐    ┌───────────────-──┐    ┌─────────────────┐
│  React Frontend │    │ FastAPI Backend  │    │ AI & Data Layer │
│                 │    │                  │    │                 │
│ • Chat Interface│◄──►│ • LangGraph Agent│◄──►│ • OpenAI GPT-4o │
│ • Role Selection│    │ • Security Tools │    │ • ChromaDB (RAG)│
│ • Transparency  │    │ • RBAC & Audit   │    │ • Tavily Search │
│ • Web Controls  │    │ • DLP & Masking  │    │ • LangSmith     │
└─────────────────┘    └─────────────────-┘    └─────────────────┘
```

## 🧠 AI Agent Architecture

### LangGraph ReAct Agent
The core AI system uses LangGraph's ReAct (Reasoning + Acting) pattern:

```python
# Agent with autonomous tool selection
agent = create_react_agent(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    tools=[PolicySearchTool(), LogQueryTool(), WebSearchTool()],
    checkpointer=SqliteSaver()  # Persistent memory
)
```

### Tool Selection Strategy
The agent autonomously decides which tool to use based on user queries:

1. **PolicySearchTool**: RAG-powered document search for security policies
2. **LogQueryTool**: Role-based security log analysis 
3. **WebSearchTool**: Real-time threat intelligence via Tavily

## 🔍 Retrieval-Augmented Generation (RAG)

### Document Processing Pipeline
```python
# 1. Document chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# 2. Embedding generation
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 3. Vector storage with metadata
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
```

### Role-Based Document Access
Documents are tagged with accessible roles during ingestion:
```python
# Metadata for role-based filtering
metadata = {
    "source": filename,
    "accessible_roles": "security,sales"  # or "security" only
}
```

## 🛡️ Security Implementation

### 1. Role-Based Access Control (RBAC)
```json
{
  "roles": {
    "security": {
      "accessible_files": ["all_handbooks"],
      "permissions": ["read_logs", "read_all_handbooks", "web_search"]
    },
    "sales": {
      "accessible_files": ["basic_policies_only"],
      "permissions": ["read_basic_policies", "web_search"]
    }
  }
}
```

### 2. Data Loss Prevention (DLP)
Automatic detection and masking of sensitive data:
```python
DLP_PATTERNS = [
    DLPPattern(name="username", pattern=r'\b[a-zA-Z]+\.[a-zA-Z]+\b', preserve_chars=2),
    DLPPattern(name="ipv4_address", pattern=r'\b(?:\d{1,3}\.){3}\d{1,3}\b', preserve_chars=4),
    DLPPattern(name="api_key", pattern=r'\b[a-zA-Z0-9]{32,}\b', preserve_chars=4),
    # ... additional patterns
]
```

### 3. Prompt Injection Defense
Pattern-based detection of malicious inputs:
```python
MALICIOUS_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"you\s+are\s+now\s+in\s+developer\s+mode",
    r"disregard\s+your\s+training"
]
```

### 4. Security Transparency System
Complete decision tracking with `SecurityDecisionTracker`:
```python
class SecurityDecisionTracker:
    def __init__(self):
        self.steps = []              # Processing steps
        self.security_checks = {}    # Security validations  
        self.tool_usage = {}        # Tool selection decisions
        self.data_sources = []      # Information sources
        self.access_decisions = {}  # RBAC decisions
        self.confidence_scores = [] # Confidence metrics
```

## 📊 Observability & Monitoring

### LangSmith Integration
Enhanced observability with comprehensive tracing:
```python
@traceable(name="security_assistant_chat")
async def chat_endpoint(request: ChatRequest):
    # Full request tracing with LangSmith
    ...

@traceable(name="policy_search_tool")  
def _run(self, query: str, user_role: str = "sales") -> str:
    # Tool-level tracing
    ...
```

### Audit Logging
Comprehensive action tracking with DLP integration:
```python
class AuditLogEntry(BaseModel):
    timestamp: datetime
    user_role: str
    action: str
    query: str
    tool_used: Optional[str]
    result: str
    langsmith_trace_id: Optional[str]  # Trace correlation
    langsmith_project: Optional[str]
```

## 💬 Conversation Memory

### Persistent Multi-turn Conversations
```python
# SQLite-based conversation persistence
checkpointer = SqliteSaver(
    sqlite3.connect("conversations.db", check_same_thread=False)
)

# Conversation threading
config = {"configurable": {"thread_id": conversation_id}}
response = agent.invoke({"messages": messages}, config=config)
```

## 🎨 Frontend Architecture

### React Component Structure
```
src/
├── App.tsx              # Main chat interface
├── App.css              # Styling with markdown support
└── index.tsx            # Application entry point
```

### Markdown Rendering
Custom markdown renderer for rich AI responses:
```tsx
const renderMarkdown = (text: string) => {
    // Handle headers, bold text, lists, code blocks
    // Support for security documentation formatting
}
```

### Security Transparency UI
Expandable transparency sections showing:
- Processing summary (steps, timing, security checks)
- Security status validation  
- Tool selection justification
- Confidence assessment with limitations
- Data sources and access control decisions

## 📁 Data Architecture

### Mock Data Structure
```
mock_data/
├── handbooks/
│   ├── phishing_response.md         # Accessible: security, sales
│   ├── general_security_policy.md   # Accessible: security, sales  
│   └── incident_escalation.md       # Accessible: security only
├── logs/
│   └── security_logs.csv           # Role-based filtering
└── rbac_config.json                # Permission definitions
```

### Vector Database (ChromaDB)
- **Embedding Model**: OpenAI text-embedding-3-small
- **Chunk Size**: 1000 characters with 200 overlap
- **Metadata Filtering**: Role-based document access
- **Persistence**: Local file-based storage

## 🔧 API Design

### Core Endpoints
```python
POST /chat                    # Main AI conversation
GET  /health                  # System health + LangSmith status  
GET  /audit-logs             # Security audit trail (security role only)
POST /new-conversation       # Start new conversation thread
POST /conversation-history   # Retrieve conversation history
GET  /available-documents    # List accessible documents by role
GET  /dlp-status            # DLP monitoring (security role only)
```

### Request/Response Models
```python
class ChatRequest(BaseModel):
    message: str
    user_role: str
    conversation_id: Optional[str] = None
    web_search_enabled: Optional[bool] = True

class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []
    tool_calls: List[str] = []
    transparency: Optional[Dict[str, Any]] = None
```

## 🚀 Deployment Architecture

### Development Stack
- **Python 3.12+** with virtual environment
- **Node.js 16+** for React frontend
- **FastAPI** with Uvicorn ASGI server
- **React** with Create React App
- **SQLite** for conversation persistence
- **ChromaDB** for vector storage

### Production Considerations
- **Containerization**: Docker support ready
- **Load Balancing**: Stateless backend design
- **Database**: PostgreSQL for production audit logs
- **Vector Store**: Scalable vector database (Pinecone, Weaviate)
- **Monitoring**: LangSmith + application monitoring
- **Security**: Environment-based secrets management

## 🎯 Security Design Principles

### 1. Defense in Depth
Multiple security layers: prompt injection detection, RBAC, DLP, audit logging

### 2. Principle of Least Privilege  
Role-based access with minimal necessary permissions

### 3. Transparency by Design
Complete audit trail and decision explanations for accountability

### 4. Privacy Protection
Automatic sensitive data detection and masking

### 5. Zero Trust Architecture
Every request validated and logged with full traceability

## 📈 Performance Characteristics

### Response Times
- **Policy Search**: ~2-3 seconds (RAG + LLM)
- **Log Query**: ~1-2 seconds (CSV processing)
- **Web Search**: ~3-5 seconds (external API + LLM)

### Scalability
- **Concurrent Users**: 10-50 (development), 100+ (production with scaling)
- **Document Corpus**: 1000+ documents supported
- **Conversation History**: Unlimited with SQLite/PostgreSQL

### Resource Usage
- **Memory**: ~500MB base + embeddings cache
- **Storage**: ~100MB + conversation history
- **CPU**: Moderate during LLM inference

---

**This architecture provides enterprise-grade security, observability, and scalability while maintaining simplicity for development and demonstration purposes.**
