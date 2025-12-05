import { useState, useRef, useEffect } from "react";
import PropTypes from "prop-types";
import "./ChatSidebar.css";
import { API_BASE_URL } from "../config.js";

export default function ChatSidebar({ open, onClose, reportContext }) {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (open && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 300);
    }
  }, [open]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isStreaming) return;

    const userMessage = inputValue.trim();
    setInputValue("");
    
    // Add user message
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    
    // Add empty assistant message that will be streamed
    setMessages((prev) => [...prev, { role: "assistant", content: "", isStreaming: true }]);
    setIsStreaming(true);

    // Log context for debugging
    console.log("Sending chat with context:", reportContext);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage,
          report_context: reportContext,
          chat_history: messages.slice(-10),
        }),
      });

      if (!response.ok) {
        throw new Error("Chat request failed");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantResponse = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.text) {
                assistantResponse += data.text;
                setMessages((prev) => {
                  const newMessages = [...prev];
                  const lastIdx = newMessages.length - 1;
                  if (lastIdx >= 0 && newMessages[lastIdx].role === "assistant") {
                    newMessages[lastIdx] = {
                      ...newMessages[lastIdx],
                      content: assistantResponse,
                    };
                  }
                  return newMessages;
                });
              }
              
              if (data.done) {
                setMessages((prev) => {
                  const newMessages = [...prev];
                  const lastIdx = newMessages.length - 1;
                  if (lastIdx >= 0) {
                    newMessages[lastIdx] = {
                      ...newMessages[lastIdx],
                      isStreaming: false,
                    };
                  }
                  return newMessages;
                });
              }
              
              if (data.error) {
                throw new Error(data.error);
              }
            } catch (parseError) {
              // Skip malformed JSON
            }
          }
        }
      }
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => {
        const newMessages = [...prev];
        const lastIdx = newMessages.length - 1;
        if (lastIdx >= 0 && newMessages[lastIdx].role === "assistant") {
          newMessages[lastIdx] = {
            role: "assistant",
            content: "Sorry, I encountered an error. Please try again.",
            isStreaming: false,
          };
        }
        return newMessages;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <>
      {/* Overlay */}
      <div 
        className={`chat-overlay ${open ? "visible" : ""}`} 
        onClick={onClose}
      />
      
      {/* Sidebar */}
      <aside className={`chat-sidebar ${open ? "open" : ""}`}>
        <header className="chat-header">
          <div className="chat-title">
            <span className="chat-icon">💬</span>
            <h3>Chat with eMMMy</h3>
          </div>
          <div className="chat-header-actions">
            <button 
              type="button" 
              className="clear-chat-btn" 
              onClick={clearChat}
              title="Clear chat"
            >
              🗑️
            </button>
            <button 
              type="button" 
              className="close-sidebar-btn" 
              onClick={onClose}
            >
              ✕
            </button>
          </div>
        </header>

        <div className="chat-messages">
          {messages.length === 0 ? (
            <div className="chat-empty-state">
              {/* <div className="empty-icon">🤖</div> */}
              <h4>Hi, I'm eMMMy!</h4>
              <p>
                Ask me anything about your optimization results, budget allocation, 
                or marketing strategy recommendations.
              </p>
              
              {/* Show optimization summary */}
              {reportContext?.target_sales && (
                <div className="context-status">
                  <p className="context-ready">
                    ✅ Target: ${reportContext.target_sales.toLocaleString()} | 
                    Predicted: ${reportContext.predicted_sales?.toLocaleString() || 'N/A'} | 
                    Optimizer: {reportContext.optimizer_type?.toUpperCase() || 'N/A'}
                  </p>
                </div>
              )}

              <div className="suggested-questions">
                <p className="suggested-label">Try asking:</p>
                <button 
                  type="button"
                  onClick={() => setInputValue("What channels should I prioritize?")}
                >
                  What channels should I prioritize?
                </button>
                <button 
                  type="button"
                  onClick={() => setInputValue("Explain the optimization results")}
                >
                  Explain the optimization results
                </button>
                <button 
                  type="button"
                  onClick={() => setInputValue("How can I improve ROI?")}
                >
                  How can I improve ROI?
                </button>
              </div>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className={`chat-message ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === "user" ? "👤" : "🤖"}
                </div>
                <div className="message-content">
                  {msg.content}
                  {msg.isStreaming && <span className="typing-cursor">▋</span>}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input-container">
          <textarea
            ref={inputRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask eMMMy about your optimization..."
            disabled={isStreaming}
            rows={1}
          />
          <button
            type="button"
            className="send-btn"
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isStreaming}
          >
            {isStreaming ? (
              <span className="send-spinner" />
            ) : (
              "➤"
            )}
          </button>
        </div>
      </aside>
    </>
  );
}

ChatSidebar.propTypes = {
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  reportContext: PropTypes.object,
};

