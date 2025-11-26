"""
AI Agent Automation with LangGraph - Complete Implementation
Customer Support Automation System using Groq API
"""
import os
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import operator
from datetime import datetime
import json
import logging

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    query: str
    category: str
    retrieved_data: str
    response: str
    quality_score: int
    timestamp: str
    metadata: dict
    
# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Configuration for AI Agent System"""
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MODEL_NAME = "llama-3.3-70b-versatile"
    TEMPERATURE = 0.7
    MAX_TOKENS = 1000
    
    #Knowledge Base Simulation
    KNOWLEDGE_BASE = {
        'ORDER_STATUS': {
            "data": "Order tracking system connected. Typical delivery: 3-5 business days.",
            "context": "Customer inquiring about order status"
        },
        'REFUND': {
            "data": "Refund policy: 30-day return window. Full refund with receipt. Processing time: 5-7 business days.",
            "context": "Customer requesting refund information"
        },
        'SHIPPING': {
            "data": "Shipping options: Standard (5-7 days, $5.99), Express (2-3 days, $12.99), Overnight ($24.99).",
            "context": "Customer inquiring about shipping"
        },
        'POLICY': {
            "data": "Return policy: 30 days, original condition. Warranty: 1 year manufacturer warranty on electronics.",
            "context": "Customer asking about policies"
        },
        'TECHNICAL_SUPPORT': {
            "data": "Technical support available 24/7. Common issues: password reset, account access, product setup.",
            "context": "Customer needs technical assistance"
        },
        'GENERAL': {
            "data": "General customer support. Business hours: Mon-Fri 9AM-6PM EST. Contact: support@company.com",
            "context": "General inquiry"
        }
    }
    
# ============================================================================
# LLM INITIALIZATION
# ============================================================================
def initialize_llm():
    """Initialize the Groq LLM"""
    try:
        llm = ChatGroq(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            groq_api_key=Config.GROQ_API_KEY
        )
        logger.info("LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialized LLM: {e}")
        raise
    
# ============================================================================
# AGENT NODES
# ============================================================================
def classify_query_node(state: AgentState) -> AgentState:
    """
    Node 1: Classify the incoming query into categories
    """
    logger.info(f"Classifying query: {state['query']}")
    
    try:
        llm = initialize_llm()
        
        prompt = f"""Analyze this customer support query and classify it into ONE of this categories:
        - ORDER_STATUS: Questions about order tracking, delivery status
        - REFUND: Refund requests, return inquiries
        - SHIPPING: Shipping methods, costs, delivery times
        - POLICY: Return policies, warranties, terms
        - TECHNICAL_SUPPORT: Technical issues, troubleshooting
        - GENERAL: Everything else
        
        Query: {state['query']}
        
        Responde with ONLY the category name, nothing else."""
        
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        category = response.content.strip()
        
        logger.info(f"Query classified as: {category}")
        
        return {
            **state,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "node": "classifier",
                "status": "completed"
            }
        }
    except Exception as e:
        logger.error(f"Error in classify_query_node: {e}")
        return {
            **state,
            "category": "GENERAL",
            "metadata": {
                "node": "classifier",
                "status": "error",
                "error": str(e)
            }
        }
        
def retrieve_data_node(state: AgentState) -> AgentState:
    """
    Node 2: Retrieve relevant data based on classification
    """
    logger.info(f"Retrieving data for category: {state['category']}")
    
    try:
        category = state.get('category', "GENERAL")
        
        # Retrieve from knowledge base
        knowledge = Config.KNOWLEDGE_BASE.get(category, Config.KNOWLEDGE_BASE["GENERAL"])
        retrieved_data = knowledge["data"]
        
        logger.info(f"Data retrieved successfully for {category}")
        
        # Return updated state
        return {
            **state, # Unpack the previous state
            "retrieved_data": retrieved_data,
            "metadata": {
                **state.get("metadata", {}),
                "retriever": "completed",
                "data_source": "knowledge_base"
            }
        }
    except Exception as e:
        logger.error(f"Error retrieved_data_node: {e}")
        return {
            **state, 
            "retrieved_data": "Error retrieving data",
            "metadata": {
                **state.get("metadata", {}),
                "retriever": "error"
            }
        }
        
def generate_response_node(state: AgentState) -> AgentState:
    """
    Node 3: Generate a helpful response using LLM
    """
    logger.info("Generating response")
    
    try:
        llm = initialize_llm()
        
        system_prompt = """You are a professional and friendly customer support agent. Your goal is to provide helpful, acurate and empathetic reponses to customer queries. Keep responses concise but complete. Be professional yet warm"""
        
        user_prompt = f"""Based on this information: {state['retrieved_data']}, Provide a helpful response to this customer query: {state['query']}. Make the response personal, professionl and actionable."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        generated_response = response.content.strip()
        
        logger.info(f"Response generated successfully")
        
        return {
            **state,
            "response": generated_response,
            "metadata": {
                **state.get("metadata", {}),
                "generator": "completed"
            }
        }
    except Exception as e:
        logger.error(f"Error in generate_response_node: {e}")
        return {
            **state,
            "response": "I appologize, but I'm having trouble generating a response. Please try again.",
            "metadata": {
                **state.get("metadata", {}),
                "generator": "error"
            }
        }
        
def quality_check_node(state: AgentState) -> AgentState:
    """
    Node 4: Check the quality of the generated response.
    """
    logger.info("Performing quality check")
    
    try:
        llm = initialize_llm()
        
        prompt = f"""Evaluate this customer support response on a scale of 1-10:
        
        Original Query: {state['query']}
        Response: {state['response']}
        
        Consider:
        - Relevance to the query
        - Helpfulness and completeness
        - Professionalism and tone
        - Clarity and conciseness
        
        Respond with ONLY a number from 1-10, nothing else
        """
        
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        
        try:
            quality_score = int(response.content.strip())
        except ValueError:
            quality_score = 7 # Default score if parsing fails
        
        logging.info(f"Quality score: {quality_score}/10")
        
        return {
            **state,
            "quality_score": quality_score,
            "metadata": {
                **state.get("metadata", {}),
                "quality_checker": "completed",
                "final_status": "success"
            }
        }
    except Exception as e:
        logger.error(f"Error in quality_check_node: {e}")
        return {
            **state,
            "quality_score": 0,
            "metadata": {
                **state.get("metadata", {}),
                "quality_checker": "error",
            }
        }
        
def should_regenerate(state: AgentState) -> Literal["regenerate", "end"]:
    """
    Conditional Edge: Decide if response needs regeneration.
    """
    quality_score = state.get("quality_score", 0)
    
    # If quality score is too low, regenerate (max 1 retry to avoid loops)
    retry_count = state.get("metadata", {}).get("retry_count", 0)
    
    if quality_score < 6 and retry_count < 1:
        logger.info(f"Quality score too low ({quality_score}/10), regenerating...")
        return "regenerate"
    return "end"

# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================
def create_agent_workflow():
    """
    Create and compile the Langgraph workflow.
    """
    logger.info("Creating agent workflow graph")
    
    #Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("classifier", classify_query_node)
    workflow.add_node("retriever", retrieve_data_node)
    workflow.add_node("generator", generate_response_node)
    workflow.add_node("quality_checker", quality_check_node)
    
    # Define the flow
    workflow.set_entry_point("classifier")
    workflow.add_edge("classifier", "retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", "quality_checker")
    
    # Conditional edge for regeneration
    workflow.add_conditional_edges(
        "quality_checker",
        should_regenerate,
        {
            "regenerate": "generator",
            "end": END
        }
    )
    
    #Compile the graph
    app = workflow.compile()
    
    logger.info("Workflow compiled successfully")
    return app

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def run_agent(query: str):
    """
    Run the agent workflow with a given query
    """
    logger.info(f"Starting agent workflow for query: {query}")
    
    # Create the workflow
    app = create_agent_workflow()
    
    initial_state = {
        "query": query,
        "category": "",
        "retrieved_data": "",
        "response": "",
        "quality_score": 0,
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "retry_count": 0
        }
    }
    
    # Run the workflow
    try:
        final_state = app.invoke(initial_state)
        logger.info("Workflow completed successfully")
        return final_state
    except Exception as e:
        logger.error(f"Error running workflow: {e}")
        raise
    
def display_results(result: AgentState):
    """
    Display the results in a formatted way.
    """
    print("\n" + "="*80)
    print("AI Agent Automation Results")
    print("="*80)
    print(f"\nOriginal Query: {result['query']}")
    print(f"\nCategory: {result['category']}")
    print(f"\nRetrieved Data: {result['retrieved_data']}")
    print(f"\nGenerated Response: {result['response']}/10")
    print(f"\nQuality Score: {result['quality_score']}")
    print(f"\nTimestamp: {result['timestamp']}")
    print(f"\nMetadata: {json.dumps(result.get("metadata", {}), indent=2)}")
    print("\n" + "="*80 + "\n")
    

# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Set your Groq API key
    # os.environ["GROQ_API_KEY"] = "your_api_key_here"
    
    # Check if API key is set
    if not Config.GROQ_API_KEY:
        print("❌ Error: GROQ_API_KEY not set!")
        print("Please set your Groq API key:")
        print("export GROQ_API_KEY='your_api_key_here'")
        exit(1)
        
    # Example queries to test
    test_queries = [
        "What's the status of my order #12345?",
        "I need to return a defective product and get a refund",
        "How long does standard shipping take?",
        "What is your return policy?",
        "I'm having trouble logging into my account"
    ]
    
    print("\n🤖 AI Agent Automation with LangGraph")
    print("Using Groq API with Llama 3.3 70B\n")
    
    for i, query in enumerate(test_queries, 1):
        print("\n" + "="*80)
        print(f"TEST CASE {i}/{len(test_queries)}")
        print("\n" + "="*80)
        
        try:
            result = run_agent(query)
            display_results(result)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
        
        if i < len(test_queries):
            input("\nPress Enter to continue to the next test case...")
            
    print("\n🎉 All test cases completed!\n")
    print("\n To use with your own queries, call: run_agent('your_query_here')")