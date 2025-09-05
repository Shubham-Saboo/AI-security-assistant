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
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_tavily import TavilySearch

# LangSmith imports for enhanced observability
from langsmith import traceable
from langsmith.wrappers import wrap_openai

# Environment and utilities
from dotenv import load_dotenv
import chromadb
import sqlite3
import re

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
transparency_tracker = None

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

class NewConversationRequest(BaseModel):
    user_role: str = Field(..., description="User role: security or sales")

class ConversationHistoryRequest(BaseModel):
    conversation_id: str = Field(..., description="Conversation ID to get history for")
    user_role: str = Field(..., description="User role: security or sales")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant response")
    sources: List[str] = Field(default=[], description="Source documents used")
    tool_calls: List[str] = Field(default=[], description="Tools called")
    transparency: Optional[Dict[str, Any]] = Field(default=None, description="Security transparency explanation")

class AuditLogEntry(BaseModel):
    timestamp: datetime
    user_role: str
    action: str
    query: str
    tool_used: Optional[str]
    result: str
    langsmith_trace_id: Optional[str] = Field(default=None, description="LangSmith trace ID for observability")
    langsmith_project: Optional[str] = Field(default=None, description="LangSmith project name")
    
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

def is_security_related_question(text: str, web_search_enabled: bool) -> bool:
    """Check if the question is security-related - ONLY security questions allowed regardless of web search status"""
    text_lower = text.lower()
    
    # Security-related keywords - only questions containing these are allowed
    security_keywords = [
        'security', 'phishing', 'malware', 'virus', 'firewall', 'password', 'authentication',
        'authorization', 'encryption', 'incident', 'breach', 'vulnerability', 'threat',
        'attack', 'hacker', 'cybersecurity', 'policy', 'compliance', 'audit', 'log',
        'access', 'permission', 'role', 'vpn', 'ssl', 'tls', 'certificate', 'antivirus',
        'backup', 'recovery', 'forensics', 'intrusion', 'ddos', 'ransomware', 'trojan',
        'spyware', 'social engineering', 'two-factor', '2fa', 'mfa', 'zero trust',
        'endpoint', 'network security', 'data protection', 'privacy', 'gdpr', 'compliance',
        'risk assessment', 'penetration test', 'security assessment', 'cve', 'patch',
        'escalation', 'incident response', 'login attempt', 'failed login', 'user access'
    ]
    
    # Check if any security keywords are present
    return any(keyword in text_lower for keyword in security_keywords)

# ========================
# DATA LOSS PREVENTION (DLP)
# ========================

class DLPPattern:
    """DLP pattern definition with masking strategy"""
    def __init__(self, name: str, pattern: str, mask_char: str = "*", preserve_chars: int = 0):
        self.name = name
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.mask_char = mask_char
        self.preserve_chars = preserve_chars

# Define DLP patterns for different types of sensitive data
DLP_PATTERNS = [
    # Personal Identifiers
    DLPPattern(
        name="username",
        pattern=r'\b[a-zA-Z]+\.[a-zA-Z]+\b',  # john.doe format
        preserve_chars=2
    ),
    DLPPattern(
        name="user_id",
        pattern=r'\bu\d{3,}\b',  # u001, u002 format
        preserve_chars=1
    ),
    DLPPattern(
        name="email",
        pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        preserve_chars=3
    ),
    
    # Network & System Information
    DLPPattern(
        name="ipv4_address",
        pattern=r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        preserve_chars=4  # Show first octet: 192.***.***.**
    ),
    DLPPattern(
        name="internal_ip",
        pattern=r'\b(?:10\.|172\.(?:1[6-9]|2[0-9]|3[01])\.|192\.168\.)\d{1,3}\.\d{1,3}\b',
        preserve_chars=3
    ),
    
    # Security Credentials
    DLPPattern(
        name="api_key",
        pattern=r'\b[a-zA-Z0-9]{32,}\b',  # Long alphanumeric strings
        preserve_chars=4
    ),
    DLPPattern(
        name="token",
        pattern=r'\b(?:sk-|pk_|tvly-)[a-zA-Z0-9_-]{20,}\b',  # API key prefixes
        preserve_chars=6
    ),
    
    # Financial Data
    DLPPattern(
        name="credit_card",
        pattern=r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        preserve_chars=4
    ),
    DLPPattern(
        name="ssn",
        pattern=r'\b\d{3}-?\d{2}-?\d{4}\b',
        preserve_chars=0  # Fully mask SSNs
    ),
]

def mask_sensitive_data(text: str, user_role: str = "unknown") -> tuple[str, list]:
    """
    Apply DLP masking to sensitive data in text
    
    Args:
        text: Input text to scan and mask
        user_role: User role for role-based masking policies
        
    Returns:
        Tuple of (masked_text, detected_patterns)
    """
    if not text:
        return text, []
    
    masked_text = text
    detected_patterns = []
    
    for dlp_pattern in DLP_PATTERNS:
        matches = dlp_pattern.pattern.finditer(text)
        
        for match in matches:
            original_value = match.group()
            
            # Apply role-based masking policies
            if should_mask_for_role(dlp_pattern.name, user_role):
                masked_value = apply_masking(original_value, dlp_pattern)
                masked_text = masked_text.replace(original_value, masked_value)
                
                detected_patterns.append({
                    "type": dlp_pattern.name,
                    "original_length": len(original_value),
                    "position": match.start(),
                    "masked": True
                })
    
    return masked_text, detected_patterns

def should_mask_for_role(data_type: str, user_role: str) -> bool:
    """
    Determine if data should be masked based on user role
    """
    # Role-based masking policies
    role_policies = {
        "security": {
            # Security team sees more data but still masks credentials
            "api_key": True,
            "token": True,
            "credit_card": True,
            "ssn": True,
            "email": False,  # Security can see emails
            "username": False,  # Security can see usernames
            "user_id": False,
            "ipv4_address": False,  # Security needs to see IPs
            "internal_ip": False
        },
        "sales": {
            # Sales team has more restricted access
            "api_key": True,
            "token": True,
            "credit_card": True,
            "ssn": True,
            "email": True,  # Mask emails from sales
            "username": True,  # Mask usernames from sales
            "user_id": True,
            "ipv4_address": True,  # Mask IPs from sales
            "internal_ip": True
        }
    }
    
    # Default to strict masking for unknown roles
    default_policy = {data_type: True for data_type in [p.name for p in DLP_PATTERNS]}
    
    return role_policies.get(user_role, default_policy).get(data_type, True)

def apply_masking(value: str, pattern: DLPPattern) -> str:
    """
    Apply masking to a specific value based on the pattern configuration
    """
    if pattern.preserve_chars == 0:
        # Fully mask
        return pattern.mask_char * len(value)
    
    if len(value) <= pattern.preserve_chars:
        # If value is too short, mask everything
        return pattern.mask_char * len(value)
    
    # Preserve first N characters, mask the rest
    preserved = value[:pattern.preserve_chars]
    masked_portion = pattern.mask_char * (len(value) - pattern.preserve_chars)
    
    return preserved + masked_portion

def log_dlp_event(user_role: str, data_types: list, context: str):
    """Log DLP masking events for security monitoring"""
    if data_types:
        dlp_entry = AuditLogEntry(
            timestamp=datetime.now(),
            user_role=user_role,
            action="dlp_masking_applied",
            tool_used="dlp_system",
            query=context[:50] + "..." if len(context) > 50 else context,
            result=f"Masked {len(data_types)} sensitive data types: {', '.join(set([d['type'] for d in data_types]))}",
            langsmith_project=os.getenv("LANGSMITH_PROJECT", "security-assistant") if os.getenv("LANGSMITH_TRACING", "false").lower() == "true" else None
        )
        audit_log.append(dlp_entry)
        logger.info(f"DLP: Masked sensitive data for {user_role}: {[d['type'] for d in data_types]}")

# ========================
# SECURITY TRANSPARENCY SYSTEM
# ========================

class SecurityDecisionTracker:
    """Track AI decision-making process for security transparency"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset tracking for new request"""
        self.steps = []
        self.security_checks = {}
        self.tool_usage = {}
        self.data_sources = []
        self.confidence_scores = {}
        self.access_decisions = {}
        self.processing_time = 0
        self.start_time = datetime.now()
    
    def add_step(self, step_name: str, description: str, status: str = "completed", details: Dict = None):
        """Add a processing step"""
        self.steps.append({
            "step": step_name,
            "description": description,
            "status": status,
            "timestamp": datetime.now(),
            "details": details or {}
        })
    
    def add_security_check(self, check_name: str, result: bool, reason: str = ""):
        """Record security check results"""
        self.security_checks[check_name] = {
            "passed": result,
            "reason": reason,
            "timestamp": datetime.now()
        }
    
    def add_tool_usage(self, tool_name: str, reason: str, confidence: float = 0.0, results_count: int = 0):
        """Record tool usage and reasoning"""
        self.tool_usage[tool_name] = {
            "reason": reason,
            "confidence": confidence,
            "results_count": results_count,
            "timestamp": datetime.now()
        }
    
    def add_data_source(self, source: str, relevance: str, access_level: str):
        """Record data sources used"""
        self.data_sources.append({
            "source": source,
            "relevance": relevance,
            "access_level": access_level,
            "timestamp": datetime.now()
        })
    
    def add_access_decision(self, resource: str, granted: bool, reason: str):
        """Record access control decisions"""
        self.access_decisions[resource] = {
            "granted": granted,
            "reason": reason,
            "timestamp": datetime.now()
        }
    
    def finalize(self):
        """Finalize tracking and calculate metrics"""
        self.processing_time = (datetime.now() - self.start_time).total_seconds()
    
    def get_explanation(self, user_role: str) -> Dict[str, Any]:
        """Generate simple explanation for the user"""
        self.finalize()
        
        # Generate simple transparency explanation
        explanation_text = self._get_simple_explanation()
        
        return {
            "explanation": explanation_text
        }
    
    def _get_simple_explanation(self) -> str:
        """Generate a simple one-sentence explanation of tool usage and data sources"""
        if not self.tool_usage and not self.data_sources:
            return "Response generated using general AI knowledge without external tools or data sources."
        
        # Get tool information
        tools_used = list(self.tool_usage.keys())
        data_sources = [source["source"] for source in self.data_sources]
        
        if tools_used and data_sources:
            tool_name = tools_used[0]  # Primary tool used
            source_name = data_sources[0]  # Primary data source
            
            tool_display = {
                "policy_search": "Policy Search tool",
                "log_query": "Log Query tool", 
                "web_search": "Web Search tool"
            }.get(tool_name, tool_name)
            
            return f"Response generated using {tool_display} based on {source_name}."
        elif tools_used:
            tool_name = tools_used[0]
            tool_display = {
                "policy_search": "Policy Search tool",
                "log_query": "Log Query tool",
                "web_search": "Web Search tool" 
            }.get(tool_name, tool_name)
            return f"Response generated using {tool_display}."
        else:
            return "Response generated using general AI knowledge."
    

# Global transparency tracker will be initialized in startup

def log_audit_entry(user_role: str, action: str, query: str, tool_used: str = None, result: str = "", trace_id: str = None):
    """Log action for audit purposes with DLP masking and LangSmith integration"""
    # Apply DLP masking to query and result
    masked_query, query_dlp_patterns = mask_sensitive_data(query, user_role)
    masked_result, result_dlp_patterns = mask_sensitive_data(result, user_role)
    
    # Log DLP events if sensitive data was detected
    all_dlp_patterns = query_dlp_patterns + result_dlp_patterns
    if all_dlp_patterns:
        log_dlp_event(user_role, all_dlp_patterns, f"Query: {query[:50]}, Result: {result[:50]}")
    
    # Get LangSmith information if available
    langsmith_project = os.getenv("LANGSMITH_PROJECT", "security-assistant") if os.getenv("LANGSMITH_TRACING", "false").lower() == "true" else None
    
    # Create audit entry with masked data and LangSmith trace info
    entry = AuditLogEntry(
        timestamp=datetime.now(),
        user_role=user_role,
        action=action,
        query=masked_query,
        tool_used=tool_used,
        result=masked_result[:200] + "..." if len(masked_result) > 200 else masked_result,
        langsmith_trace_id=trace_id,
        langsmith_project=langsmith_project
    )
    audit_log.append(entry)
    
    # Enhanced logging with LangSmith info
    log_msg = f"AUDIT: {user_role} - {action} - {tool_used}"
    if trace_id:
        log_msg += f" - Trace: {trace_id}"
    logger.info(log_msg)

# ========================
# DOCUMENT PROCESSING & CHROMADB
# ========================

def initialize_embeddings_and_llm():
    """Initialize OpenAI embeddings and LLM with LangSmith tracing"""
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
    
    # Check for LangSmith configuration
    langsmith_tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    if langsmith_tracing and langsmith_api_key:
        logger.info("LangSmith tracing enabled - enhanced observability active")
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "security-assistant")
    else:
        logger.info("LangSmith tracing disabled or API key not provided")
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )
    
    # Use LangSmith wrapped OpenAI client if tracing is enabled
    if langsmith_tracing and langsmith_api_key:
        from openai import OpenAI
        openai_client = wrap_openai(OpenAI(api_key=openai_api_key))
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        logger.info("OpenAI LLM initialized with LangSmith tracing")
    else:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        logger.info("OpenAI LLM initialized without LangSmith tracing")
    
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
    description: str = "Search security policies and handbooks. Use this when users ask about security procedures, policies, or company specific guidance."
    
    @traceable(name="policy_search_tool")
    def _run(self, query: str, user_role: str = "sales") -> str:
        """Search for relevant policy documents"""
        global transparency_tracker
        try:
            transparency_tracker.add_tool_usage("policy_search", f"User asked about security policies/procedures: '{query[:50]}...'")
            docs = search_documents_by_role(query, user_role, k=3)
            
            if not docs:
                transparency_tracker.add_tool_usage("policy_search", "No matching documents found", confidence=0.0, results_count=0)
                return "No relevant policy documents found for your query."
            
            # Add transparency tracking for successful search
            transparency_tracker.add_tool_usage("policy_search", f"Found {len(docs)} relevant policy documents", confidence=0.9, results_count=len(docs))
            
            # Format response
            response = "Found relevant policy information:\n\n"
            sources = []
            
            for i, doc in enumerate(docs, 1):
                source_file = doc.metadata.get("source_file", "unknown")
                sources.append(source_file)
                
                # Add data source tracking
                accessibility = "role-filtered" if user_role in doc.metadata.get("accessible_roles", "").split(",") else "unrestricted"
                transparency_tracker.add_data_source(source_file, "high", accessibility)
                transparency_tracker.add_access_decision(source_file, True, f"Document accessible to {user_role} role")
                
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
    
    @traceable(name="log_query_tool")
    def _run(self, query: str, user_role: str = "sales") -> str:
        """Query security logs with role-based filtering"""
        global transparency_tracker
        try:
            transparency_tracker.add_tool_usage("log_query", f"User requested log analysis: '{query[:50]}...'")
            
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
                transparency_tracker.add_tool_usage("log_query", "No matching log entries found", confidence=0.0, results_count=0)
                return "No matching log entries found for your query."
            
            # Add transparency tracking for successful search
            transparency_tracker.add_tool_usage("log_query", f"Found {len(filtered_df)} security log entries", confidence=0.9, results_count=len(filtered_df))
            transparency_tracker.add_data_source("security_logs.csv", "high", f"role-filtered ({user_role})")
            transparency_tracker.add_access_decision("security_logs.csv", True, f"Log access granted to {user_role} role")
            
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
    description: str = "Search the web for real-time information on any security related topic including threat intelligence, security news, CVE vulnerabilities, cybersecurity trends, business information, and general knowledge. Use this when users need current, up-to-date information from the internet."
    
    @traceable(name="web_search_tool")
    def _run(self, query: str, user_role: str = "security") -> str:
        """Search the web for real-time information on any topic"""
        global transparency_tracker
        try:
            transparency_tracker.add_tool_usage("web_search", f"User requested web search: '{query[:50]}...'")
            
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
                transparency_tracker.add_tool_usage("web_search", "No search results found", confidence=0.0, results_count=0)
                return "No search results found for your query. Try rephrasing or being more specific."
            
            # Add transparency tracking for successful search
            results_count = len(search_results.get("results", []))
            transparency_tracker.add_tool_usage("web_search", f"Found {results_count} web search results", confidence=0.8, results_count=results_count)
            transparency_tracker.add_data_source("Tavily Web Search", "real-time", "public web sources")
            transparency_tracker.add_access_decision("web_search", True, f"Web search enabled for {user_role} role")
            
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
                action="web_search",
                query=query,
                tool_used="web_search", 
                result=f"Found {len(results)} web search results"
            )
            
            response += "\n**⚠️ Security Note:** Always verify threat intelligence from multiple sources and consult your incident response procedures for any confirmed threats."
            
            return response
            
        except Exception as e:
            logger.error(f"Error in threat intelligence search: {e}")
            return f"Error performing threat intelligence search: {str(e)}"

# ========================
# AGENT SETUP
# ========================

@traceable(name="create_security_agent")
def create_security_agent(web_search_enabled: bool = True):
    """Create LangChain agent with security tools and persistent memory"""
    
    # Create base tools
    tools = [
        PolicySearchTool(),
        LogQueryTool()
    ]
    
    # Conditionally add web search tool
    if web_search_enabled:
        tools.append(WebSearchTool())
    
    # Create persistent memory using SQLite
    memory_db_path = "conversations.db"
    try:
        # Try to use SQLite for persistent memory
        conn = sqlite3.connect(memory_db_path, check_same_thread=False)
        memory = SqliteSaver(conn)
    except Exception as e:
        logger.warning(f"Failed to create SQLite checkpointer, falling back to in-memory: {e}")
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
    global transparency_tracker
    logger.info("Starting Security Assistant API...")
    
    try:
        # Initialize transparency tracker
        transparency_tracker = SecurityDecisionTracker()
        
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
    langsmith_status = {
        "enabled": os.getenv("LANGSMITH_TRACING", "false").lower() == "true",
        "api_key_configured": bool(os.getenv("LANGSMITH_API_KEY")),
        "project": os.getenv("LANGSMITH_PROJECT", "security-assistant")
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "components": {
            "llm": llm is not None,
            "embeddings": embeddings is not None,
            "vector_store": vector_store is not None,
            "rbac_config": bool(rbac_config),
            "langsmith": langsmith_status
        }
    }

@app.post("/chat", response_model=ChatResponse)
@traceable(name="security_assistant_chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint with AI assistant"""
    
    # Initialize transparency tracking
    global transparency_tracker
    transparency_tracker.reset()
    transparency_tracker.add_step("request_received", f"Processing query from {request.user_role} role")
    
    # Security: Check for prompt injection
    transparency_tracker.add_step("prompt_injection_check", "Scanning for malicious input patterns")
    if detect_prompt_injection(request.message):
        transparency_tracker.add_security_check("prompt_injection", False, "Malicious patterns detected")
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
    
    transparency_tracker.add_security_check("prompt_injection", True, "No malicious patterns found")
    
    # Security: Check if question is security-related when web search is disabled
    if not is_security_related_question(request.message, request.web_search_enabled):
        transparency_tracker.add_security_check("security_topic", False, "Non-security question with web search disabled")
        
        decline_message = ("I'm a specialized security assistant focused on helping with security policies and logs. "
                          "I can't answer general questions like that. Instead, I can help you with security procedures, "
                          "incident response, or analyze security logs. What security topic can I assist you with?")
        
        log_audit_entry(
            user_role=request.user_role,
            action="non_security_question_declined",
            query=request.message,
            result=decline_message
        )
        
        # Generate transparency explanation for declined request
        transparency_explanation = transparency_tracker.get_explanation(request.user_role)
        
        return ChatResponse(
            response=decline_message,
            sources=[],
            tool_calls=[],
            transparency=transparency_explanation
        )
    
    transparency_tracker.add_security_check("security_topic", True, "Security-related question or web search enabled")
    
    # Validate user role
    transparency_tracker.add_step("rbac_validation", "Validating user role and permissions")
    if request.user_role not in rbac_config["roles"]:
        transparency_tracker.add_security_check("rbac_validation", False, f"Invalid role: {request.user_role}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid user role"
        )
    
    transparency_tracker.add_security_check("rbac_validation", True, f"Valid role: {request.user_role}")
    transparency_tracker.add_access_decision("system_access", True, f"Role {request.user_role} authorized")
    
    try:
        # Create agent with web search setting
        transparency_tracker.add_step("agent_creation", f"Creating AI agent with web_search={request.web_search_enabled}")
        agent = create_security_agent(web_search_enabled=request.web_search_enabled)
        
        # Configuration for conversation
        config = {
            "configurable": {
                "thread_id": request.conversation_id or "default"
            },
            "recursion_limit": 10
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
- Web search is currently disabled. You can ONLY answer questions about:
  * Security policies and procedures (use policy_search tool)
  * Security logs and events (use log_query tool)
- For ANY other topics (current events, general knowledge, non-security questions), politely decline and redirect to security topics."""
        
        system_prompt = f"""You are an expert security assistant specialized EXCLUSIVELY in enterprise cybersecurity topics. You ONLY answer security-related questions.

User Role: {request.user_role}

AVAILABLE TOOLS:
{tools_description}

ABSOLUTE RESTRICTIONS:
- You ONLY answer questions related to cybersecurity, information security, or IT security
- ALL non-security questions must be politely declined regardless of circumstances
- You are NOT a general assistant - you are a specialized security consultant ONLY

DECISION FRAMEWORK:
1. **SECURITY TOPIC VERIFICATION**: Every question must be security-related
   - ✅ ALLOWED: security policies, procedures, incidents, logs, threats, vulnerabilities, compliance, authentication, access control, cybersecurity
   - ❌ FORBIDDEN: weather, sports, politics, general knowledge, entertainment, personal questions, non-security topics

2. **DATA SOURCE SELECTION** (for security questions only):
   - **Policy Search**: Questions about OUR security policies, procedures, incident response, compliance requirements
   - **Log Query**: Questions about OUR security events, user activities, login attempts, system logs, incidents
   - **Web Search**: Current security threats, recent CVEs, latest security news{web_search_guidance}

3. **RESPONSE REQUIREMENTS**:
   - ALWAYS use the appropriate tool and cite the specific data source
   - Pass user_role="{request.user_role}" to all tools
   - If information is not available, explain what you CAN help with

CRITICAL INSTRUCTIONS:
- For NON-SECURITY questions: ALWAYS decline politely and redirect to security topics
- For SECURITY questions: Use the appropriate tool to provide accurate information
- NEVER provide answers about non-security topics under any circumstances
- You are a SECURITY-ONLY assistant - stay strictly within your domain

EXAMPLES:
- "How do I handle phishing?" → ✅ Use policy_search
- "Show me failed login attempts" → ✅ Use log_query  
- "Latest ransomware threats" → ✅ Use web_search or policy_search
- "What's our password policy?" → ✅ Use policy_search
- "Who is the president?" → ❌ DECLINE (not security-related)
- "What's the weather?" → ❌ DECLINE (not security-related)
- "How to cook pasta?" → ❌ DECLINE (not security-related)

You are a specialized security assistant - ONLY answer security-related questions."""

        # Check if this is a new conversation by trying to get current state
        try:
            current_state = agent.get_state(config)
            is_new_conversation = not bool(current_state.values.get("messages", []))
        except Exception:
            is_new_conversation = True
        
        # For new conversations, we need to include the system prompt
        # For continuing conversations, we just add the user message and let LangGraph handle the rest
        if is_new_conversation:
            # Start new conversation with system prompt
            input_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message}
            ]
        else:
            # Continue existing conversation - just add user message
            # LangGraph will automatically maintain the conversation history
            input_messages = [
                {"role": "user", "content": request.message}
            ]
        
        # Run the agent - it will handle conversation state automatically
        transparency_tracker.add_step("ai_processing", "Executing AI agent with selected tools")
        response = agent.invoke(
            {"messages": input_messages},
            config=config
        )
        
        # Extract the response
        assistant_message = response["messages"][-1].content
        transparency_tracker.add_step("response_generated", f"AI generated {len(assistant_message)} character response")
        
        # Apply DLP masking to the assistant's response
        transparency_tracker.add_step("dlp_processing", "Applying data loss prevention scanning")
        masked_response, dlp_patterns = mask_sensitive_data(assistant_message, request.user_role)
        
        # Log DLP masking if sensitive data was detected in response
        if dlp_patterns:
            transparency_tracker.add_security_check("dlp_masking", True, f"Masked {len(dlp_patterns)} sensitive data patterns")
            log_dlp_event(request.user_role, dlp_patterns, f"AI Response: {assistant_message[:50]}")
        else:
            transparency_tracker.add_security_check("dlp_masking", True, "No sensitive data detected")
        
        # Extract tool calls from response (simplified)
        tool_calls = []
        sources = []
        
        # Generate transparency explanation
        transparency_explanation = transparency_tracker.get_explanation(request.user_role)
        
        # Log the interaction (using original message for audit, but the response will be logged masked)
        log_audit_entry(
            user_role=request.user_role,
            action="chat_query",
            query=request.message,
            result=assistant_message[:100] + "..." if len(assistant_message) > 100 else assistant_message
        )
        
        return ChatResponse(
            response=masked_response,  # Return the DLP-masked response
            sources=sources,
            tool_calls=tool_calls,
            transparency=transparency_explanation
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )

@app.post("/new-conversation")
async def start_new_conversation(request: NewConversationRequest):
    """Start a new conversation and return conversation ID"""
    try:
        # Generate unique conversation ID
        conversation_id = str(uuid.uuid4())
        
        # Log the new conversation
        log_audit_entry(
            user_role=request.user_role,
            action="new_conversation_started",
            query="",
            result=f"New conversation started with ID: {conversation_id}"
        )
        
        return {
            "conversation_id": conversation_id,
            "message": "New conversation started successfully"
        }
        
    except Exception as e:
        logger.error(f"Error starting new conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting new conversation: {str(e)}"
        )

@app.post("/conversation-history")
async def get_conversation_history(request: ConversationHistoryRequest):
    """Get conversation history for a specific conversation ID"""
    try:
        # Create agent to access conversation state
        agent = create_security_agent()
        
        # Configuration for the specific conversation
        config = {
            "configurable": {
                "thread_id": request.conversation_id
            }
        }
        
        # Get the conversation state
        try:
            state = agent.get_state(config)
            messages = state.values.get("messages", [])
            
            # Format messages for frontend
            formatted_messages = []
            for msg in messages:
                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                    # Skip system messages for cleaner history
                    if msg.type != "system":
                        formatted_messages.append({
                            "role": msg.type,
                            "content": msg.content,
                            "timestamp": datetime.now().isoformat()  # Simplified timestamp
                        })
            
            return {
                "conversation_id": request.conversation_id,
                "messages": formatted_messages,
                "total_messages": len(formatted_messages)
            }
            
        except Exception as e:
            # Conversation doesn't exist or is empty
            return {
                "conversation_id": request.conversation_id,
                "messages": [],
                "total_messages": 0
            }
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting conversation history: {str(e)}"
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

@app.get("/dlp-status")
async def get_dlp_status(user_role: str = "security"):
    """Get DLP masking statistics and patterns (Security role only)"""
    if user_role != "security":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Security role required for DLP monitoring"
        )
    
    # Count DLP events in audit log
    dlp_events = [log for log in audit_log if log.action == "dlp_masking_applied"]
    
    # Statistics
    total_dlp_events = len(dlp_events)
    recent_dlp_events = [log for log in dlp_events if 
                        (datetime.now() - log.timestamp).total_seconds() < 3600]  # Last hour
    
    # Pattern analysis
    pattern_stats = {}
    for event in dlp_events:
        # Extract pattern types from result
        if "Masked" in event.result:
            patterns = event.result.split(": ")[-1] if ": " in event.result else ""
            for pattern in patterns.split(", "):
                pattern = pattern.strip()
                if pattern:
                    pattern_stats[pattern] = pattern_stats.get(pattern, 0) + 1
    
    return {
        "dlp_monitoring": {
            "total_events": total_dlp_events,
            "recent_events_1h": len(recent_dlp_events),
            "pattern_statistics": pattern_stats,
            "supported_patterns": [p.name for p in DLP_PATTERNS],
            "role_policies": {
                "security": "Relaxed masking for operational needs",
                "sales": "Strict masking for data protection"
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
