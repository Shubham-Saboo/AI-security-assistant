import React, { useState, useRef, useEffect } from 'react';
import './App.css';

// Simple markdown renderer for basic formatting
const renderMarkdown = (text: string) => {
  return text.split('\n').map((line, index) => {
    // Handle headers (### Header)
    if (line.startsWith('### ')) {
      return (
        <h3 key={index} className="markdown-h3">
          {line.replace('### ', '')}
        </h3>
      );
    }
    
    // Handle headers (## Header)
    if (line.startsWith('## ')) {
      return (
        <h2 key={index} className="markdown-h2">
          {line.replace('## ', '')}
        </h2>
      );
    }
    
    // Handle numbered lists (1. Item)
    if (/^\d+\.\s/.test(line)) {
      const content = line.replace(/^\d+\.\s/, '');
      return (
        <div key={index} className="markdown-list-item numbered">
          <span className="list-number">{line.match(/^\d+/)?.[0]}.</span>
          <span className="list-content">{renderInlineMarkdown(content)}</span>
        </div>
      );
    }
    
    // Handle bullet points (- Item) with various indentation levels
    if (line.match(/^\s*-\s/)) {
      const leadingSpaces = line.match(/^(\s*)/)?.[1].length || 0;
      const indentLevel = Math.floor(leadingSpaces / 3); // Every 3 spaces = 1 indent level
      const content = line.replace(/^\s*-\s/, '');
      return (
        <div key={index} className={`markdown-list-item bullet indent-${indentLevel}`}>
          <span className="bullet-point">•</span>
          <span className="list-content">{renderInlineMarkdown(content)}</span>
        </div>
      );
    }
    
    // Handle empty lines
    if (line.trim() === '') {
      return <br key={index} />;
    }
    
    // Handle regular paragraphs
    return (
      <p key={index} className="markdown-paragraph">
        {renderInlineMarkdown(line)}
      </p>
    );
  });
};

// Render inline markdown (bold, italic, code)
const renderInlineMarkdown = (text: string) => {
  // Handle **bold** text
  let parts = text.split(/(\*\*[^*]+\*\*)/g);
  
  return parts.map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return (
        <strong key={index} className="markdown-bold">
          {part.slice(2, -2)}
        </strong>
      );
    }
    
    // Handle `code` text
    if (part.includes('`')) {
      const codeParts = part.split(/(`[^`]+`)/g);
      return codeParts.map((codePart, codeIndex) => {
        if (codePart.startsWith('`') && codePart.endsWith('`')) {
          return (
            <code key={`${index}-${codeIndex}`} className="markdown-code">
              {codePart.slice(1, -1)}
            </code>
          );
        }
        return codePart;
      });
    }
    
    return part;
  });
};

// Types
interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  transparency?: any; // Transparency explanation from backend
}

interface ChatResponse {
  response: string;
  sources: string[];
  tool_calls: string[];
  transparency?: any; // Security transparency explanation
}

const App: React.FC = () => {
  // State
  const [userRole, setUserRole] = useState<'security' | 'sales' | ''>('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [webSearchEnabled, setWebSearchEnabled] = useState(true);
  const [conversationId, setConversationId] = useState<string>('');
  
  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Focus input when role is selected
  useEffect(() => {
    if (userRole && inputRef.current) {
      inputRef.current.focus();
    }
  }, [userRole]);

  // API call to backend
  const sendMessage = async (message: string) => {
    if (!message.trim() || !userRole) return;

    setIsLoading(true);
    setError('');

    // Add user message
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: message,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          user_role: userRole,
          conversation_id: conversationId || 'default-session',
          web_search_enabled: webSearchEnabled
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to send message');
      }

      const data: ChatResponse = await response.json();

      // Add assistant message with transparency data
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response,
        timestamp: new Date(),
        transparency: data.transparency // Include transparency explanation
      };
      setMessages(prev => [...prev, assistantMessage]);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      console.error('Chat error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Start a new conversation
  const startNewConversation = async () => {
    try {
      const response = await fetch('http://localhost:8000/new-conversation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_role: userRole
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setConversationId(data.conversation_id);
        setMessages([]);
        setError('');
        console.log('New conversation started:', data.conversation_id);
      }
    } catch (err) {
      console.error('Failed to start new conversation:', err);
    }
  };

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(inputMessage);
  };

  // Sample queries for different roles
  const allSampleQueries = {
    security: {
      basic: [
        "How should I handle a suspected phishing email?",
        "Show me today's failed login attempts from the logs",
        "What's the escalation path for a security breach?"
      ],
      webSearch: [
        "Research the latest CVE vulnerabilities affecting web applications",
        "Find recent threat intelligence about ransomware attacks",
        "What are the current cybersecurity trends and threats?"
      ]
    },
    sales: {
      basic: [
        "How should I handle a suspicious email?",
        "What's our policy for customer data protection?",
        "Show me recent login activity for sales team",
        "What should I do if I suspect a phishing attempt?"
      ],
      webSearch: [
        "Search for recent news about our industry competitors",
        "Find the latest trends in cybersecurity for sales presentations"
      ]
    }
  };

  // Get filtered sample queries based on web search toggle
  const getSampleQueries = (role: 'security' | 'sales') => {
    const queries = [...allSampleQueries[role].basic];
    if (webSearchEnabled) {
      queries.push(...allSampleQueries[role].webSearch);
    }
    return queries;
  };

  // Role Selection Screen
  if (!userRole) {
    return (
      <div className="app">
        <div className="role-selection">
          <div className="role-card">
            <h1>🛡️ Security Assistant</h1>
            <p>AI-powered security incident knowledge assistant</p>
            
            <div className="role-buttons">
              <button 
                className="role-btn security-btn"
                onClick={() => setUserRole('security')}
              >
                <div className="role-icon">👨‍💼</div>
                <div className="role-info">
                  <h3>Security Team</h3>
                  <p>Full access to policies, logs, and incident procedures</p>
                </div>
              </button>
              
              <button 
                className="role-btn sales-btn"
                onClick={() => setUserRole('sales')}
              >
                <div className="role-icon">👩‍💼</div>
                <div className="role-info">
                  <h3>Sales Team</h3>
                  <p>Access to basic security policies and sales-related logs</p>
                </div>
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Chat Interface
  return (
    <div className="app">
      <div className="chat-container">
        {/* Header */}
        <div className="chat-header">
          <div className="header-left">
            <h2>🛡️ Security Assistant</h2>
            <span className={`role-badge ${userRole}`}>
              {userRole === 'security' ? '👨‍💼 Security Team' : '👩‍💼 Sales Team'}
            </span>
          </div>
          <div className="header-controls">
            <div className="web-search-toggle">
              <label className="toggle-label">
                <span className="toggle-text">🌐 Web Search</span>
                <input
                  type="checkbox"
                  checked={webSearchEnabled}
                  onChange={(e) => setWebSearchEnabled(e.target.checked)}
                  className="toggle-checkbox"
                />
                <span className="toggle-slider"></span>
              </label>
            </div>
            <button 
              className="new-chat-btn"
              onClick={startNewConversation}
              title="Start a new conversation"
            >
              💬 New Chat
            </button>
            <button 
              className="change-role-btn"
              onClick={() => {
                setUserRole('');
                setMessages([]);
                setError('');
                setConversationId('');
              }}
            >
              Change Role
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="messages-container">
          {messages.length === 0 && (
            <div className="welcome-message">
              <h3>Welcome! Try these sample queries:</h3>
              <div className="sample-queries">
                {getSampleQueries(userRole).map((query, index) => (
                  <button
                    key={index}
                    className="sample-query"
                    onClick={() => sendMessage(query)}
                  >
                    {query}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className={`message ${message.role}`}>
              <div className="message-content">
                <div className="message-text">
                  {renderMarkdown(message.content)}
                </div>
                
                {/* Transparency explanation for assistant messages */}
                {message.role === 'assistant' && message.transparency && (
                  <div className="transparency-section">
                    <details className="transparency-details">
                      <summary className="transparency-summary">
                        🔍 Why did I answer this way? (Security Transparency)
                      </summary>
                      <div className="transparency-content">
                        <div className="transparency-item">
                          <strong>📊 Processing Summary:</strong>
                          <ul>
                            <li>Steps: {message.transparency.processing_summary?.total_steps || 'N/A'}</li>
                            <li>Processing time: {message.transparency.processing_summary?.processing_time_ms || 'N/A'}ms</li>
                            <li>Security checks passed: {message.transparency.processing_summary?.security_checks_passed || 'N/A'}</li>
                            <li>Tools used: {message.transparency.processing_summary?.tools_used || 'N/A'}</li>
                          </ul>
                        </div>
                        
                        <div className="transparency-item">
                          <strong>🛡️ Security Status:</strong>
                          <span className={`security-status ${message.transparency.security_analysis?.overall_status?.toLowerCase()}`}>
                            {message.transparency.security_analysis?.overall_status || 'Unknown'}
                          </span>
                        </div>
                        
                        {message.transparency.tool_justification?.tools_selected?.length > 0 && (
                          <div className="transparency-item">
                            <strong>🔧 Tools Used:</strong>
                            <ul>
                              {message.transparency.tool_justification.tools_selected.map((tool: any, idx: number) => (
                                <li key={idx}>
                                  <strong>{tool.tool}</strong>: {tool.justification} 
                                  {tool.confidence !== 'N/A' && <span className="confidence"> (Confidence: {tool.confidence})</span>}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        
                        {message.transparency.confidence_assessment && (
                          <div className="transparency-item">
                            <strong>📈 Confidence Assessment:</strong>
                            <div>Overall: {message.transparency.confidence_assessment.overall_confidence}</div>
                            <div>Basis: {message.transparency.confidence_assessment.basis}</div>
                            {message.transparency.confidence_assessment.limitations && (
                              <div>Limitations: {message.transparency.confidence_assessment.limitations}</div>
                            )}
                          </div>
                        )}
                      </div>
                    </details>
                  </div>
                )}
                
                <div className="message-time">
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="message assistant">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <form className="chat-input" onSubmit={handleSubmit}>
          <input
            ref={inputRef}
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Ask about security policies, incidents, or logs..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !inputMessage.trim()}>
            {isLoading ? '⏳' : '🚀'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default App;