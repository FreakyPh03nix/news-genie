import nltk
from nltk.tokenize import word_tokenize
import time
import requests
import os
from google.colab import userdata # Keep for local testing if needed, but will be removed for deployment
from tavily import TavilyClient
import re
from typing import TypedDict
from langgraph.graph import StateGraph, END
import streamlit as st

# NLTK Downloads
# These usually run once, but for a script, they might need a check or be pre-downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except nltk.downloader.DownloadError:
    nltk.download('punkt_tab')
# print("'punkt' tokenizer data downloaded successfully.") # Removed print for clean app.py

# API Key Initialization
# In a deployed Streamlit app, these would typically be loaded from environment variables
# or Streamlit secrets, not userdata.get(). For this step, I'll keep userdata.get()
# as it reflects the notebook state, but will add comments for deployment.
NEWS_API_KEY = userdata.get('GNEWS_API_KEY')
TAVILY_API_KEY = userdata.get('TAVILY_API_KEY')

# For NewsAPI: NEWS_API_BASE_URL = "https://newsapi.org/v2/everything"
# For GNews API: NEWS_API_BASE_URL = "https://gnews.io/api/v4/search"
NEWS_API_BASE_URL = "https://gnews.io/api/v4/search"

# Initialize the Tavily client with the API key
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

def preprocess_text(text):
    """
    Tokenizes and normalizes the input text.
    Converts to lowercase and removes punctuation (for simplicity).
    """
    tokens = word_tokenize(text.lower())
    # Remove non-alphabetic tokens
    normalized_tokens = [word for word in tokens if word.isalpha()]
    return normalized_tokens

def recognize_intent(user_input):
    """
    Analyzes user input to classify the intent as either 'general_query' or 'news_request'.
    """
    processed_input = preprocess_text(user_input)
    lower_user_input = user_input.lower()

    # Define single-word news keywords
    news_single_keywords = ["news", "headlines"]

    # Define multi-word news phrases based on the Chatbot Scope and Requirements Definition
    # and observations from previous testing
    news_phrases_to_match = [
        "latest updates", "current events", "breaking news", "today's news",
        "recent developments", "what's happening", "tell me the news",
        "give me today's headlines", "what are the latest stories",
        "what is happening in the world"
    ]

    # Check for single-word keyword matches in processed tokens
    for token in processed_input:
        if token in news_single_keywords:
            return "news_request"

    # Check for multi-word phrase matches in the original lowercased input
    for phrase in news_phrases_to_match:
        if phrase in lower_user_input:
            return "news_request"

    return "general_query"

def fetch_news(query):
    """
    Fetches news articles from a news API based on the query.
    """
    if not NEWS_API_KEY:
        return "News API key is not set. Please provide a valid API key."

    if not NEWS_API_BASE_URL:
        return "News API base URL is not set. Please provide a valid URL."

    try:
        params = {
            'q': query,
            'lang': 'en',
            'country': 'us', # Or other relevant country codes
            'max': 5, # Number of articles to fetch
            'apikey': NEWS_API_KEY
        }
        response = requests.get(NEWS_API_BASE_URL, params=params)
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()

        articles = data.get('articles', [])
        if not articles:
            return "Sorry, I couldn't find any news for that query."

        news_output = []
        for i, article in enumerate(articles):
            title = article.get('title', 'No Title')
            description = article.get('description', 'No Description')
            url = article.get('url', '#')
            news_output.append(f"{i+1}. {title}\n   {description}\n   Read more: {url}\n")

        return "\n".join(news_output)

    except requests.exceptions.RequestException as e:
        return f"Error fetching news: {e}"
    except ValueError as e:
        return f"Error parsing API response: {e}"

def handle_news_request(user_input, category="General"):
    """
    Handles a news-specific request by extracting keywords from the user input
    and fetching relevant news articles, optionally filtered by category.
    """
    # Define phrases that indicate a news request but should be removed from the query
    news_request_phrases = [
        "tell me the news about",
        "give me today's headlines about",
        "what are the latest stories on",
        "what are the latest updates on",
        "latest news on",
        "news about",
        "headlines about",
        "current events about",
        "breaking news about",
        "today's news on",
        "recent developments on",
        "what's happening with",
        "tell me the news",
        "give me today's headlines",
        "what are the latest stories",
        "what is happening in the world",
        "news",
        "headlines",
        "latest updates",
        "current events",
        "breaking news",
        "today's news",
        "recent developments",
        "what's happening"
    ]

    query = user_input.lower()

    # Remove news request phrases from the query
    for phrase in news_request_phrases:
        if phrase in query:
            query = query.replace(phrase, "").strip()

    # Further clean the query by removing punctuation (except spaces) and extra spaces
    query = re.sub(r'[^a-zA-Z0-9\s]', '', query)
    query = re.sub(r'\s+', ' ', query).strip()

    # Append category to query if it's not 'General' and query is not empty
    if category != "General" and category.strip() != "":
        if query:
            query = f"{query} {category}"
        else:
            query = category # If only category was selected and no specific query

    # If the query becomes empty after removing phrases and cleaning, set a default
    if not query:
        query = "top stories"

    return fetch_news(query)

def fetch_web_results(query):
    """
    Fetches web search results using the Tavily API.
    """
    if not TAVILY_API_KEY:
        return "Tavily API key is not set. Please provide a valid API key."

    try:
        response = tavily_client.search(query=query, search_depth="advanced", max_results=5)

        results = response.get('results', [])
        if not results:
            return "Sorry, I couldn't find any relevant web results for that query using Tavily."

        # Synthesize a main answer from the first result's content or a concise summary
        synthesized_answer = response.get('answer', 'No summary available.')

        web_results_output = []
        if synthesized_answer and synthesized_answer != 'No summary available.':
            web_results_output.append(f"Here's what I found: {synthesized_answer}")
        else:
            # Fallback to the first result's content if no overall summary is provided
            first_result_content = results[0].get('content', 'No content available.')
            if first_result_content and len(first_result_content) > 100: # Take a snippet
                web_results_output.append(f"Here's a snippet from a top result: {first_result_content[:200]}...")
            elif first_result_content:
                web_results_output.append(f"Here's what I found: {first_result_content}")
            else:
                web_results_output.append("Here are some top results:")

        web_results_output.append("Top results:")

        for i, item in enumerate(results):
            title = item.get('title', 'No Title')
            url = item.get('url', '#')
            # Tavily 'content' can be long, so we take a snippet for description
            content_snippet = item.get('content', 'No description available.')
            if len(content_snippet) > 150:
                content_snippet = content_snippet[:150] + '...'

            web_results_output.append(f"{i+1}. {title}\n   {content_snippet}\n   Read more: {url}\n")

        return "\n".join(web_results_output)

    except Exception as e:
        return f"Error fetching web results from Tavily: {e}"

def handle_general_query(user_input):
    """
    Processes a general query and provides a response based on a simple rule-based system,
    with a fallback to web search for unknown queries.
    """
    lower_input = user_input.lower()

    # Simple keyword-based rule system
    if "what is ai" in lower_input or "define ai" in lower_input:
        return "AI stands for Artificial Intelligence, which is the simulation of human intelligence processes by machines, especially computer systems."
    elif "capital of france" in lower_input:
        return "The capital of France is Paris."
    elif "who invented the light bulb" in lower_input:
        return "Thomas Edison is widely credited with inventing the practical incandescent light bulb."
    elif "how to tie a shoelace" in lower_input:
        return "To tie a shoelace, make two 'bunny ears' with the laces, cross them over, and then tuck one under the other before pulling tight."
    elif "your name" in lower_input or "who are you" in lower_input:
        return "I am an AI assistant designed to help with your queries."
    elif "hello" in lower_input or "hi" in lower_input:
        return "Hello! How can I assist you today?"
    elif "how are you" in lower_input:
        return "I am an AI, so I don't have feelings, but I'm ready to help you!"
    else:
        # Fallback to web search if no rule-based answer is found
        # print(f"No direct answer found for '{user_input}'. Attempting web search...") # Removed print for clean app.py
        web_search_result = fetch_web_results(user_input)
        if web_search_result:
            return web_search_result
        else:
            return "I'm not sure how to answer that general query. Could you please rephrase it or ask something else?"

# Graph State Definition
class GraphState(TypedDict):
    """
    Represents the state of our graph. Used to pass information between nodes.
    """
    user_input: str
    intent: str
    response: str
    news_category: str # Added for news category selection

# LangGraph Node Functions
def intent_recognition_node(state: GraphState) -> GraphState:
    """
    Node to recognize the intent of the user input.
    """
    user_input = state["user_input"]
    intent = recognize_intent(user_input)
    return {"intent": intent}

def general_query_node(state: GraphState) -> GraphState:
    """
    Node to handle general queries.
    """
    user_input = state["user_input"]
    response = handle_general_query(user_input)
    return {"response": response}

def news_request_node(state: GraphState) -> GraphState:
    """
    Node to handle news requests.
    """
    user_input = state["user_input"]
    # Pass the selected news category to handle_news_request
    selected_news_category = state.get("news_category", "General")
    response = handle_news_request(user_input, selected_news_category)
    time.sleep(2) # Preserve the delay
    return {"response": response}

# LangGraph Workflow Setup and Compilation
def route_intent(state: GraphState):
    """
    Conditional edge based on intent.
    """
    if state["intent"] == "general_query":
        return "general_query_handler"
    elif state["intent"] == "news_request":
        return "news_request_handler"
    else:
        return "general_query_handler" # Fallback if intent is not recognized or an unexpected value

# Build the graph
workflow = StateGraph(GraphState)

# Add nodes to the graph
workflow.add_node("intent_recognizer", intent_recognition_node)
workflow.add_node("general_query_handler", general_query_node)
workflow.add_node("news_request_handler", news_request_node)

# Set the entry point
workflow.set_entry_point("intent_recognizer")

# Add conditional edges from the intent recognizer
workflow.add_conditional_edges(
    "intent_recognizer",
    route_intent,
    {
        "general_query_handler": "general_query_handler",
        "news_request_handler": "news_request_handler",
    },
)

# Set end points for the handlers
workflow.add_edge("general_query_handler", END)
workflow.add_edge("news_request_handler", END)

# Compile the graph
app = workflow.compile()

# Chatbot Response Function
def chatbot_response(user_input, news_category="General"):
    """
    Main function to process user input using the LangGraph workflow.
    """
    initial_state = {"user_input": user_input, "intent": "", "response": "", "news_category": news_category}

    # Run the graph directly to get the final state
    final_state = app.invoke(initial_state)

    return final_state.get("response", "I'm sorry, something went wrong and I couldn't process your request.")

# Streamlit UI
st.title("AI Chatbot")

# News Category Selection
news_categories = ['General', 'Technology', 'Politics', 'Sports', 'Business', 'Health', 'Science']
if 'selected_news_category' not in st.session_state:
    st.session_state.selected_news_category = 'General'

st.session_state.selected_news_category = st.selectbox(
    "Select News Category:",
    news_categories,
    index=news_categories.index(st.session_state.selected_news_category),
    key="news_category_selector"
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_query := st.chat_input("Ask me anything:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate and display assistant response, passing the selected news category
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chatbot_response(user_query, st.session_state.selected_news_category)
            st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
