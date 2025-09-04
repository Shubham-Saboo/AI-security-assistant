import React, { useState, useRef, useEffect } from 'react';
import './App.css';

// Types
interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface ChatResponse {
  response: string;
  sources: string[];
  tool_calls: string[];
}

const App: React.FC = () => {
  // State
  const [userRole, setUserRole] = useState<'security' | 'sales' | ''>('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [webSearchEnabled, setWebSearchEnabled] = useState(true);
  
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
          conversation_id: 'web-session',
          web_search_enabled: webSearchEnabled
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to send message');
      }

      const data: ChatResponse = await response.json();

      // Add assistant message
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, assistantMessage]);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      console.error('Chat error:', err);
    } finally {
      setIsLoading(false);
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
              className="change-role-btn"
              onClick={() => {
                setUserRole('');
                setMessages([]);
                setError('');
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
                  {message.content.split('\n').map((line, index) => (
                    <React.Fragment key={index}>
                      {line}
                      {index < message.content.split('\n').length - 1 && <br />}
                    </React.Fragment>
                  ))}
                </div>
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