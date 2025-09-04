"""
Security Assistant Backend
A simple FastAPI application with LangChain, ChromaDB for document retrieval,
and role-based access control for security documents.
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# FastAPI and web framework
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_tavily import TavilySearch

# Environment and utilities
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Security Assistant API",
    description="AI-powered security assistant with document retrieval and role-based access",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# CONFIGURATION & GLOBALS
# ========================

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
MOCK_DATA_DIR = BASE_DIR / "mock_data"
HANDBOOKS_DIR = MOCK_DATA_DIR / "handbooks"
LOGS_DIR = MOCK_DATA_DIR / "logs"
RBAC_CONFIG_PATH = MOCK_DATA_DIR / "rbac_config.json"

# ChromaDB setup
CHROMA_DB_PATH = BASE_DIR / "chroma_db"
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

# Global variables
embeddings = None
vector_store = None
llm = None
rbac_config = {}
audit_log = []

# ========================
# PYDANTIC MODELS
# ========================

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    user_role: str = Field(..., description="User role: security or sales")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for memory")
    web_search_enabled: Optional[bool] = Field(True, description="Whether web search tool is enabled")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant response")
    sources: List[str] = Field(default=[], description="Source documents used")
    tool_calls: List[str] = Field(default=[], description="Tools called")

class AuditLogEntry(BaseModel):
    timestamp: datetime
    user_role: str
    action: str
    query: str
    tool_used: Optional[str]
    result: str
    
# ========================
# RBAC AND SECURITY
# ========================

def load_rbac_config():
    """Load role-based access control configuration"""
    global rbac_config
    try:
        with open(RBAC_CONFIG_PATH, 'r') as f:
            rbac_config = json.load(f)
        logger.info("RBAC configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load RBAC config: {e}")
        # Default minimal config
        rbac_config = {
            "roles": {
                "security": {
                    "accessible_files": ["phishing_response.md", "general_security_policy.md", "incident_escalation.md"],
                    "permissions": ["read_logs", "read_all_handbooks"]
                },
                "sales": {
                    "accessible_files": ["phishing_response.md", "general_security_policy.md"],
                    "permissions": ["read_basic_policies"]
                }
            }
        }

def check_file_access(user_role: str, filename: str) -> bool:
    """Check if user role has access to specific file"""
    if user_role not in rbac_config["roles"]:
        return False
    
    accessible_files = rbac_config["roles"][user_role]["accessible_files"]
    return filename in accessible_files

def detect_prompt_injection(text: str) -> bool:
    """Simple prompt injection detection"""
    suspicious_patterns = [
        "ignore previous instructions",
        "disregard the above",
        "act as a different ai",
        "system prompt",
        "override your instructions",
        "forget what i told you",
        "new instructions:",
        "admin mode",
        "developer mode"
    ]
    
    text_lower = text.lower()
    for pattern in suspicious_patterns:
        if pattern in text_lower:
            return True
    return False

def log_audit_entry(user_role: str, action: str, query: str, tool_used: str = None, result: str = ""):
    """Log action for audit purposes"""
    entry = AuditLogEntry(
        timestamp=datetime.now(),
        user_role=user_role,
        action=action,
        query=query,
        tool_used=tool_used,
        result=result[:200] + "..." if len(result) > 200 else result
    )
    audit_log.append(entry)
    logger.info(f"AUDIT: {user_role} - {action} - {tool_used}")

# ========================
# DOCUMENT PROCESSING & CHROMADB
# ========================

def initialize_embeddings_and_llm():
    """Initialize OpenAI embeddings and LLM"""
    global embeddings, llm
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Check for Tavily API key (optional for web search)
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if tavily_api_key:
        logger.info("Tavily API key found - web search capabilities enabled")
    else:
        logger.warning("TAVILY_API_KEY not set - web search capabilities disabled")
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        openai_api_key=openai_api_key
    )
    
    logger.info("OpenAI embeddings and LLM initialized")

def process_and_store_documents():
    """Process handbook documents, chunk them, and store in ChromaDB with role metadata"""
    global vector_store
    
    if not HANDBOOKS_DIR.exists():
        logger.error(f"Handbooks directory not found: {HANDBOOKS_DIR}")
        return
    
    # Text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Collect all documents with metadata
    all_documents = []
    
    for handbook_file in HANDBOOKS_DIR.glob("*.md"):
        try:
            # Load document
            loader = TextLoader(str(handbook_file), encoding='utf-8')
            documents = loader.load()
            
            # Split into chunks
            chunks = text_splitter.split_documents(documents)
            
            # Add role-based metadata to each chunk
            for chunk in chunks:
                filename = handbook_file.name
                
                # Determine which roles can access this document
                accessible_roles = []
                for role, config in rbac_config["roles"].items():
                    if filename in config["accessible_files"]:
                        accessible_roles.append(role)
                
                # Add metadata (ChromaDB only supports simple types)
                chunk.metadata.update({
                    "source_file": filename,
                    "accessible_roles": ",".join(accessible_roles),  # Convert list to comma-separated string
                    "file_type": "handbook",
                    "chunk_index": len(all_documents)
                })
                
                # Mark sensitive documents
                if filename == "incident_escalation.md":
                    chunk.metadata["sensitivity"] = "confidential"
                else:
                    chunk.metadata["sensitivity"] = "internal"
                
                all_documents.append(chunk)
                
            logger.info(f"Processed {len(chunks)} chunks from {filename}")
            
        except Exception as e:
            logger.error(f"Error processing {handbook_file}: {e}")
    
    # Initialize ChromaDB collection
    try:
        # Try to get existing collection or create new one
        collection_name = "security_documents"
        try:
            collection = chroma_client.get_collection(name=collection_name)
            # Clear existing data
            chroma_client.delete_collection(name=collection_name)
        except:
            pass  # Collection doesn't exist
        
        # Create new collection
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"description": "Security handbooks with role-based access"}
        )
        
        # Create vector store
        vector_store = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        
        # Add documents to vector store
        if all_documents:
            vector_store.add_documents(all_documents)
            logger.info(f"Stored {len(all_documents)} document chunks in ChromaDB")
        else:
            logger.warning("No documents found to store")
            
    except Exception as e:
        logger.error(f"Error setting up ChromaDB: {e}")
        raise

def search_documents_by_role(query: str, user_role: str, k: int = 5) -> List[Document]:
    """Search documents with role-based filtering"""
    if not vector_store:
        return []
    
    try:
        # Get all relevant documents
        docs = vector_store.similarity_search(query, k=k*2)  # Get more to filter
        
        # Filter by role
        filtered_docs = []
        for doc in docs:
            accessible_roles_str = doc.metadata.get("accessible_roles", "")
            accessible_roles = accessible_roles_str.split(",") if accessible_roles_str else []
            if user_role in accessible_roles:
                filtered_docs.append(doc)
        
        # Return top k after filtering
        return filtered_docs[:k]
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return []

# ========================
# TOOLS FOR LANGCHAIN AGENT
# ========================

class PolicySearchTool(BaseTool):
    """Tool for searching security policies and handbooks"""
    name: str = "policy_search"
    description: str = "Search security policies and handbooks. Use this when users ask about security procedures, policies, or general guidance."
    
    def _run(self, query: str, user_role: str = "sales") -> str:
        """Search for relevant policy documents"""
        try:
            docs = search_documents_by_role(query, user_role, k=3)
            
            if not docs:
                return "No relevant policy documents found for your query."
            
            # Format response
            response = "Found relevant policy information:\n\n"
            sources = []
            
            for i, doc in enumerate(docs, 1):
                source_file = doc.metadata.get("source_file", "unknown")
                sources.append(source_file)
                
                response += f"**Source {i}: {source_file}**\n"
                response += f"{doc.page_content}\n\n"
            
            # Log the search
            log_audit_entry(
                user_role=user_role,
                action="policy_search",
                query=query,
                tool_used="policy_search",
                result=f"Found {len(docs)} documents: {', '.join(sources)}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in policy search: {e}")
            return f"Error searching policies: {str(e)}"

class LogQueryTool(BaseTool):
    """Tool for querying security logs"""
    name: str = "log_query" 
    description: str = "Query security logs to find login attempts, security events, and user activities. Use this when users ask about log analysis or security events."
    
    def _run(self, query: str, user_role: str = "sales") -> str:
        """Query security logs with role-based filtering"""
        try:
            # Load logs
            log_file = LOGS_DIR / "security_logs.csv"
            if not log_file.exists():
                return "Security logs not found."
            
            df = pd.read_csv(log_file)
            
            # Role-based filtering
            if user_role == "sales":
                # Sales can only see their own department and low/medium risk
                df = df[
                    (df['department'].isin(['sales', 'marketing'])) &
                    (df['risk_score'].isin(['low', 'medium']))
                ]
            # Security role can see all logs (no filtering)
            
            # Simple query processing
            query_lower = query.lower()
            filtered_df = df.copy()
            
            # Basic filtering based on query keywords
            if "failed" in query_lower or "failure" in query_lower:
                filtered_df = filtered_df[filtered_df['result'] == 'failed']
            elif "success" in query_lower:
                filtered_df = filtered_df[filtered_df['result'] == 'success']
            
            if "login" in query_lower:
                filtered_df = filtered_df[filtered_df['action'] == 'login']
            elif "critical" in query_lower:
                filtered_df = filtered_df[filtered_df['risk_score'] == 'critical']
            elif "high" in query_lower:
                filtered_df = filtered_df[filtered_df['risk_score'].isin(['high', 'critical'])]
                
            # Limit results
            filtered_df = filtered_df.head(10)
            
            if filtered_df.empty:
                return "No matching log entries found for your query."
            
            # Format response
            response = f"Found {len(filtered_df)} log entries:\n\n"
            
            for _, row in filtered_df.iterrows():
                response += (
                    f"**{row['timestamp']}** - User: {row['username']} "
                    f"({row['department']}) - Action: {row['action']} - "
                    f"Result: {row['result']} - Risk: {row['risk_score']}\n"
                )
                if row['ip_address']:
                    response += f"  IP: {row['ip_address']}\n"
                response += "\n"
            
            # Log the search
            log_audit_entry(
                user_role=user_role,
                action="log_query", 
                query=query,
                tool_used="log_query",
                result=f"Found {len(filtered_df)} log entries"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in log query: {e}")
            return f"Error querying logs: {str(e)}"

class WebSearchTool(BaseTool):
    """Tool for real-time web search using Tavily for any information needs"""
    name: str = "web_search"
    description: str = "Search the web for real-time information on any topic including threat intelligence, security news, CVE vulnerabilities, cybersecurity trends, business information, and general knowledge. Use this when users need current, up-to-date information from the internet."
    
    def _run(self, query: str, user_role: str = "security") -> str:
        """Search the web for real-time information on any topic"""
        try:
            # Web search is now available to both Security and Sales teams
            if user_role not in ["security", "sales"]:
                return "Web search is only available to Security and Sales team members. Please contact your administrator for access."
            
            # Check if Tavily API key is available
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if not tavily_api_key:
                return "Web search capability is not configured. Please contact your administrator to enable threat intelligence features."
            
            # Initialize Tavily search with security-focused parameters
            tavily_search = TavilySearch(
                max_results=5,
                topic="general",
                include_answer=True,  # Get a quick summary
                include_raw_content=False,  # Keep response concise
                search_depth="advanced",  # More thorough search for security topics
                time_range="month"  # Focus on recent threats
            )
            
            # Enhance query based on topic
            if any(keyword in query.lower() for keyword in ['threat', 'cve', 'vulnerability', 'malware', 'security', 'attack']):
                enhanced_query = f"cybersecurity threat intelligence {query} vulnerability CVE malware"
            else:
                enhanced_query = query
            
            # Perform the search
            search_results = tavily_search.invoke({"query": enhanced_query})
            
            if not search_results or "results" not in search_results:
                return "No search results found for your query. Try rephrasing or being more specific."
            
            # Format the response
            response = "**🔍 Web Search Results:**\n\n"
            
            # Add the AI-generated answer if available
            if search_results.get("answer"):
                response += f"**Summary:** {search_results['answer']}\n\n"
            
            # Add search results
            results = search_results.get("results", [])
            if results:
                response += "**Sources:**\n"
                for i, result in enumerate(results[:5], 1):
                    response += f"{i}. **{result.get('title', 'Unknown')}**\n"
                    response += f"   URL: {result.get('url', 'N/A')}\n"
                    if result.get('content'):
                        # Truncate content for readability
                        content = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                        response += f"   Summary: {content}\n"
                    response += "\n"
            
            # Log the search
            log_audit_entry(
                user_role=user_role,
                action="threat_intelligence_search",
                query=query,
                tool_used="threat_intelligence", 
                result=f"Found {len(results)} threat intelligence sources"
            )
            
            response += "\n**⚠️ Security Note:** Always verify threat intelligence from multiple sources and consult your incident response procedures for any confirmed threats."
            
            return response
            
        except Exception as e:
            logger.error(f"Error in threat intelligence search: {e}")
            return f"Error performing threat intelligence search: {str(e)}"

# ========================
# AGENT SETUP
# ========================

def create_security_agent(web_search_enabled: bool = True):
    """Create LangChain agent with security tools"""
    
    # Create base tools
    tools = [
        PolicySearchTool(),
        LogQueryTool()
    ]
    
    # Conditionally add web search tool
    if web_search_enabled:
        tools.append(WebSearchTool())
    
    # Create agent with memory
    memory = MemorySaver()
    
    agent = create_react_agent(
        llm, 
        tools, 
        checkpointer=memory
    )
    
    return agent

# ========================
# API ENDPOINTS
# ========================

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting Security Assistant API...")
    
    try:
        # Load RBAC configuration
        load_rbac_config()
        
        # Initialize embeddings and LLM
        initialize_embeddings_and_llm()
        
        # Process and store documents
        process_and_store_documents()
        
        logger.info("Security Assistant API started successfully!")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Security Assistant API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "components": {
            "llm": llm is not None,
            "embeddings": embeddings is not None,
            "vector_store": vector_store is not None,
            "rbac_config": bool(rbac_config)
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint with AI assistant"""
    
    # Security: Check for prompt injection
    if detect_prompt_injection(request.message):
        log_audit_entry(
            user_role=request.user_role,
            action="prompt_injection_detected",
            query=request.message,
            result="Blocked suspicious input"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Potentially malicious input detected. Please rephrase your question."
        )
    
    # Validate user role
    if request.user_role not in rbac_config["roles"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid user role"
        )
    
    try:
        # Create agent with web search setting
        agent = create_security_agent(web_search_enabled=request.web_search_enabled)
        
        # Configuration for conversation
        config = {
            "configurable": {
                "thread_id": request.conversation_id or "default"
            }
        }
        
        # Prepare messages with system prompt and user query  
        tools_description = """1. policy_search: Search security policies and handbooks
2. log_query: Query security logs and events"""
        
        if request.web_search_enabled:
            tools_description += """
3. web_search: Search the web for real-time information on any topic (available to Security and Sales teams)"""
            web_search_guidance = """
- Use web_search tool for current threats, CVEs, business information, general knowledge, or any real-time data
- For threat intelligence, always remind users to verify information from multiple sources"""
        else:
            web_search_guidance = """
- Web search is currently disabled. Focus on using policy search and log query tools for available information."""
        
        system_prompt = f"""You are a helpful security assistant. Your role is to help with security-related questions using the available tools.

User Role: {request.user_role}

You have access to the following tools:
{tools_description}

Guidelines:
- Always use the appropriate tool when users ask about policies, logs, or current information
- Pass the user_role parameter to tools: user_role="{request.user_role}"{web_search_guidance}
- Be helpful and provide clear, actionable information
- If you cannot find information, suggest alternative approaches
- Do not make up information - only use what you find in the tools

Remember: You are helping with enterprise security, so be professional and accurate."""

        # Run the agent
        response = agent.invoke(
            {"messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message}
            ]},
            config=config
        )
        
        # Extract the response
        assistant_message = response["messages"][-1].content
        
        # Extract tool calls from response (simplified)
        tool_calls = []
        sources = []
        
        # Log the interaction
        log_audit_entry(
            user_role=request.user_role,
            action="chat_query",
            query=request.message,
            result=assistant_message[:100] + "..." if len(assistant_message) > 100 else assistant_message
        )
        
        return ChatResponse(
            response=assistant_message,
            sources=sources,
            tool_calls=tool_calls
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/audit-logs")
async def get_audit_logs(user_role: str = "security"):
    """Get audit logs (security role only)"""
    if user_role != "security":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Security role required"
        )
    
    # Return recent audit logs
    recent_logs = audit_log[-50:]  # Last 50 entries
    return {"logs": [log.dict() for log in recent_logs]}

@app.get("/available-documents")
async def get_available_documents(user_role: str):
    """Get list of documents accessible to user role"""
    if user_role not in rbac_config["roles"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user role"
        )
    
    accessible_files = rbac_config["roles"][user_role]["accessible_files"]
    return {"accessible_documents": accessible_files}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
