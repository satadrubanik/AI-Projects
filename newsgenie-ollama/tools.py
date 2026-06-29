import os
import subprocess
from pathlib import Path
import requests
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# --- Configuration ---
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "").strip()
if not GNEWS_API_KEY:
    print("GNews API key not configured. News requests will be unavailable until you add it to the .env file.")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b-base")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# Initialize Local LLM (Removed if you aren't using it, but kept here for completeness)
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)

# Initialize Search Tool
search_tool = DuckDuckGoSearchRun()

def fetch_news(category):
    """
    Fetches real-time news from GNews based on a category.
    """
    if not GNEWS_API_KEY:
        return "News is unavailable because the GNews API key is not configured."

    try:
        url = f"https://gnews.io/api/v4/top-headlines?topic={category}&lang=en&apikey={GNEWS_API_KEY}"
        response = requests.get(url)
        
        # Raise an exception for bad HTTP status codes (like 401 Unauthorized)
        response.raise_for_status()
        
        data = response.json()
        if "articles" in data:
            articles = data["articles"][:5] # Get top 5 articles
            news_summary = [f"- {art['title']}: {art['url']}" for art in articles]
            return "\n".join(news_summary)
        else:
            print(f"Error from GNews API: {data.get('errors', 'Unknown error')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching news: {str(e)}")
        return None
    except Exception as e:
        print(f"Failed to fetch news: {str(e)}")
        return None

def web_search(query):
    """
    Performs a general web search for current information.
    """
    try:
        result = search_tool.run(query)
        if not result:
            print("Search failed. Please check the query and try again.")
            return None
        return result
    except Exception as e:
        print(f"Search failed: {str(e)}")
        return None

def run_terminal_command(command, wait_for_completion=True):
    """Safely runs a terminal command."""
    process = subprocess.Popen(command, shell=True)
    if wait_for_completion:
        process.wait()

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Script initialized successfully.")
    
    # WARNING: Only uncomment the line below if this file is NOT named 'main.py'.
    # Otherwise, you will create an infinite loop that crashes your computer!
    # run_terminal_command(f'{sys.executable} main.py', True)
    
    # Instead, just test your functions directly:
    # print("--- Testing News ---")
    # print(fetch_news("technology"))
    
    # print("\n--- Testing Search ---")
    # print(web_search("What is the weather in Tokyo?"))