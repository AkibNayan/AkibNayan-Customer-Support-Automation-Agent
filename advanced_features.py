"""
Advanced Features for AI Agent Automation
Including memory, tool integration, and multi-agent collaboration
"""

import os
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
import json

from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# ENHANCED STATE WITH MEMORY
# ============================================================================
class EnhancedAgentState(TypedDict):
    """Enhanced state with conversation history and memory."""
    query: str
    conversation_history: List[Dict[str, str]]
    category: str
    retrieved_data: str
    response: str
    quality_score: int
    timestamp: str
    metadata: Dict[str, Any]
    user_context: Dict[str, Any]    # User preferences, where key is str and value is Any
    tools_used: List[str]
    multi_agent_results: Dict[str, Any]
    

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================
class ConversationMemory:
    """Manages conversation history and context"""
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversations: Dict[str, List[Dict]] = {}
    
    def add_message(self, user_id: str, role: str, content: str):
        """Add messages to the conversation history"""
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        self.conversations[user_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only the recent history
        if len(self.conversations[user_id]) > self.max_history:
            self.conversations[user_id] = self.conversations[user_id][-self.max_history:]
        
    def get_history(self, user_id: str) -> List[Dict]:
        """Get the conversation history for a user"""
        return self.conversations.get(user_id, [])
    
    def clear_history(self, user_id: str):
        """Clear the conversation history for a user"""
        if user_id in self.conversations:
            del self.conversations[user_id]
            
# Global Memory Instance
memory = ConversationMemory()

# ============================================================================
# CUSTOM TOOLS
# ============================================================================
@tool
def search_order_database(order_id: str) -> str:
    """Search the order database for order information"""
    # Simulate the database lookup
    mock_orders = {
        "12345": {
            "status": "In Transit",
            "expected_delivery": "2024-12-15",
            "tracking_number": "TRK123456789"
        },
        "67890": {
            "status": "Delivered",
            "expected_delivery": "2024-12-10",
            "tracking_number": "TRK987654321"
        }
    }
    
    order_info = mock_orders.get(order_id, None)
    if order_info:
        return json.dumps(order_info)
    return json.dumps({"error": "Order not found"})

@tool
def check_inventory(product_id: str) -> str:
    """Check product inventory levels"""
    # Simulate the inventory check
    mock_inventory = {
        "PROD001": {
            "in_stock": True,
            "quantity": 150
        },
        "PROD002": {
            "in_stock": False,
            "quantity": 0
        },
        "PROD003": {
            "in_stock": True,
            "quantity": 45
        }
    }
    
    inventory = mock_inventory.get(product_id, {"in_stock": False, "quantity": 0})
    # returns a JSON string
    return json.dumps(inventory)
    
@tool
def calculate_refund(order_id: str, return_reason: str) -> str:
    """Calculate refund amount based on order_id and return_reason"""
    # Simulate refund calculation
    refund_rules = {
        "defective": 1.0,   # 100% refund
        "wrong_item": 1.0,
        "not_satisfied": 0.8,   # 80% refund restocking fee
        "change_mind": 0.7
    }
    
    refund_percentage = refund_rules.get(return_reason.lower(), 0.7)
    estimated_amount = 99.99    # Mock order amount
    refund_amount = estimated_amount * refund_percentage
    
    return json.dumps({
        "refund_amount": round(refund_amount, 2),
        "refund_percentage": refund_percentage * 100,
        "processing_time": "5-7 business days"
    })
    
# ============================================================================
# TOOL EXECUTOR NODE
# ============================================================================
def tool_executor_node(state: EnhancedAgentState) -> EnhancedAgentState:
    """Execute appropriate tools based on query category"""
    category = state.get('category', '')
    query = state.get('query', '')
    tools_used = []
    tool_results = {}
    
    # Determine which tools to use
    if category == 'ORDER_STATUS' and '#' in query:
        # Extract the order id
        order_id = query.split("#")[-1].split()[0]
        result = search_order_database.invoke({"order_id": order_id})
        
        tool_results['order_info'] = result
        tools_used.append('search_order_database')
    
    elif category == 'REFUND':
        # Extract the order id
        #order_id = query.split("#")[-1].split()[0]
        #Mock order id extraction
        order_id = "12345"
        result = calculate_refund.invoke({
            "order_id": order_id,
            "return_reason": "defective"
        })
        tool_results['refund_info'] = result
        tools_used.append('calculate_refund')
        
    # Add tool results to the retrieved data 
    enhanced_data = state.get('retrieved_data', '')
    
    if tool_results:
        enhanced_data += f"\n\nTools Results: {json.dumps(tool_results, indent=2)}"
        
    return {
        **state, 
        "retrieved_data": enhanced_data,
        "tools_used": tools_used,
        "metadata": {
            **state.get("metadata", {}),
            "tools_executed": tools_used
        }
    }
         
# ============================================================================
# MULTI-AGENT COLLABORATION
# ============================================================================
class SpecializedAgent:
    """Base class for specialized agents"""
    
    def __init__(self, name: str, expertise: str):
        self.name = name
        self.expertise = expertise
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            groq_api_key=os.getenv("GROQ_API_KEY", "")
        )
        
    def process(self, query: str, context: str) -> str:
        """Process query with specialized knowledge"""
        prompt = f"""You are a {self.expertise} specialist.
        
        Context: {context}
        Query: {query}
        
        Provide a specialized response based on your expertise."""
        
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        return response.content
    
class TechnicalSupportAgent(SpecializedAgent):
    """Agent specialized in technical support"""
    
    def __init__(self):
        super().__init__("TechSupport", "technical support and troubleshooting")

class RefundSpecialistAgent(SpecializedAgent):
    """Agent specialized in refunds and returns"""
    
    def __init__(self):
        super().__init__("RefundSpecialist", "refund processing and return")
    
class OrderManagementAgent(SpecializedAgent):
    """Agent specialized in order management"""
    
    def __init__(self):
        super().__init__("OrderManagement", "order tracking and logistics")
        
        
# ============================================================================
# MULTI-AGENT COORDINATOR
# ============================================================================
def multi_agent_coordinator_node(state: EnhancedAgentState) -> EnhancedAgentState:
    """Coordinate multiple specialized agents"""
    category = state.get("category", "")
    query = state.get("query", "")
    context = state.get("retrieved_data", "")
    
    #Select appropriate specialized agent
    specialized_response = None
    agent_name = None
    
    if category == "TECHNICAL_SUPPORT":
        agent = TechnicalSupportAgent()
        specialized_response = agent.process(query, context)
        agent_name = agent.name
    
    elif category == "REFUND":
        agent = RefundSpecialistAgent()
        specialized_response = agent.process(query, context)
        agent_name = agent.name
    
    elif category == "ORDER_MANAGEMENT":
        agent = OrderManagementAgent()
        specialized_response = agent.process(query, context)
        agent_name = agent.name
    
    multi_agent_results = {
        "agent_used": agent_name,
        "specialized_response": specialized_response
    }
    
    #Use specialized response if available
    if specialized_response:
        state["response"] = specialized_response
    
    return {
        **state,
        "multi_agent_results": multi_agent_results,
        "metadata": {
            **state.get("metadata", {}),
            "multi_agent_used": agent_name is not None  #True if an agent name was provided
        }
    }
    
    
# ============================================================================
# CONTEXT-AWARE RESPONSE NODE
# ============================================================================
def context_aware_response_node(state: EnhancedAgentState) -> EnhancedAgentState:
    """Generate response with conversation history awareness."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        groq_api_key=os.getenv("GROQ_API_KEY", "")
    )
    
    # Build conversation context
    history = state.get("conversation_history", [])
    #Consider last 5 messages
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-5:]]) if history else "No previous conversation"
    
    # Get User Context
    user_context = state.get("user_context", {})
    user_info = f"User Preferences: {json.dumps(user_context)}" if user_context else ""
    
    system_prompt = f"""You are a context-aware customer support agent.
    
    Previous conversation:
    {history_text}
    {user_info}
    
    Use the conversation history to provide personalized and contextual reponses."""
    
    user_prompt = f"""Current query: {state['query']}
    
    Available information: {state['retrieved_data']}
    
    Provide a helpful, contextual response that considers the conversation history."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    return {
        **state, 
        "response": response.content,
        "metadata": {
            **state.get("metadata", {}),
            "context_aware": True
        }
    }
    

# ============================================================================
# SENTIMENT ANALYSIS NODE
# ============================================================================
def sentiment_analysis_node(state: EnhancedAgentState) -> EnhancedAgentState:
    """Analyze sentiment and adjust response accordingly."""
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        groq_api_key=os.getenv("GROQ_API_KEY", "")
    )
    
    prompt = f"""Analyze the sentiment of this customer query: {state['query']}
    
    Respond with ONLY one word: POSITIVE, NEUTRAL, NEGATIVE, or URGENT"""
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    sentiment = response.content.strip()
    
    # Adjust response tone based on sentiment
    if sentiment in ['NEGATIVE', 'URGENT']:
        # Add empathy and prioritize resolution
        adjusted_response = f"I understand your concern and I'm here to help. {state.get('response', '')}"
        state['response'] = adjusted_response
        state['metadata']['priority'] = 'high'
        
    state['metadata']['sentiment'] = sentiment
    
    return state


# ============================================================================
# ENHANCED WORKFLOW
# ============================================================================
def create_enhanced_workflow():
    """Create enhanced workflow with advanced features."""
    from langgraph_agent import (classify_query_node,
                                 retrieve_data_node,
                                 generate_response_node,
                                 quality_check_node)
    
    workflow = StateGraph(EnhancedAgentState)
    
    # Add all nodes
    workflow.add_node("classifier", classify_query_node)
    workflow.add_node("sentiment_analyzer", sentiment_analysis_node)
    workflow.add_node("retriever", retrieve_data_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("multi-agent", multi_agent_coordinator_node)
    workflow.add_node("context_aware_generator", context_aware_response_node)
    workflow.add_node("quality_checker", quality_check_node)
    
    # Define flow
    workflow.set_entry_point("classifier")
    workflow.add_edge("classifier", "sentiment_analyzer")
    workflow.add_edge("sentiment_analyzer", "retriever")
    workflow.add_edge("retriever", "tool_executor")
    workflow.add_edge("tool_executor", "multi-agent")
    workflow.add_edge("multi-agent", "context_aware_generator")
    workflow.add_edge("context_aware_generator", "quality_checker")
    workflow.add_edge("quality_checker", END)
    
    return workflow.compile()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
def run_enhanced_agent(query: str, user_id: str = "default_user"):
    """Run enhanced agent with all features"""
    # Get conversation history
    history = memory.get_history(user_id)
    
    #Mock user context 
    user_context = {
        "user_id": user_id,
        "preferred_language": "English",
        "vip_status": False
    }
    
    #Initial state
    initial_state = {
        "query": query,
        "conversation_history": history,
        "category": "",
        "retrieved_data": "",
        "response": "",
        "quality_score": 0,
        "timestamp": datetime.now().isoformat(),
        "metadata": {},
        "user_context": user_context,
        "tools_used": [],
        "multi_agent_results": {}
    }
    
    # Run the workflow
    app = create_enhanced_workflow()
    result = app.invoke(initial_state)
    
    #Update memory
    memory.add_message(user_id, "user", query)
    memory.add_message(user_id, "assistant", result["response"])
    
    return result


# ============================================================================
# EXAMPLE EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Enhanced AI Agent with Enhanced Features\n")
    
    # Test queries
    queries = [
        "What's the status of my order #12345?",
        "I received a defective product and need a refund",
        "I'm having trouble resetting my password"
    ]
    
    for query in queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}\n")
        
        result = run_enhanced_agent(query, user_id="test_user_001")
        
        print(f"Category: {result['category']}")
        print(f"Sentiment: {result['metadata'].get('sentiment', 'N/A')}")
        print(f"Tools Used: {', '.join(result['tools_used']) if result['tools_used'] else 'None'}")
        print(f"Multi-Agent: {result['multi_agent_results'].get('agent_used', 'N/A')}")
        print(f"\nResponse:\n{result['response']}")
        print(f"\nQuality Score: {result['quality_score']}/10")
        
        input("\nPress Enter to continue...")
        
    print("\n Enhanced agent demonstration completed!")