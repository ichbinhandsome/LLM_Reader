"""
Simple Ollama API client using ollama-python library.
"""
import ollama
from typing import Dict, List, Generator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """Simple client for interacting with Ollama."""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = "gemma3:1b"):
        """Initialize Ollama client."""
        self.host = host
        self.model = model
        self.client = ollama.Client(host=host)
        self.conversation_history = []
    
    def check_connection(self) -> bool:
        """Check if Ollama server is accessible."""
        try:
            self.client.list()
            return True
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """Get list of available models."""
        try:
            models = self.client.list()
            return [model['name'] for model in models['models']]
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []
    
    def generate(self, prompt: str, stream: bool = False) -> str:
        """Generate response from Ollama model."""
        try:
            if stream:
                response = ""
                for chunk in self.client.generate(model=self.model, prompt=prompt, stream=True):
                    response += chunk['response']
                return response
            else:
                response = self.client.generate(model=self.model, prompt=prompt, stream=False)
                return response['response']
        except Exception as e:
            return f"Error: {str(e)}"
    
    def chat(self, messages: List[Dict[str, str]], stream: bool = False) -> str:
        """Chat with Ollama model using conversation history."""
        try:
            if stream:
                response = ""
                for chunk in self.client.chat(model=self.model, messages=messages, stream=True):
                    if 'message' in chunk and 'content' in chunk['message']:
                        response += chunk['message']['content']
                return response
            else:
                response = self.client.chat(model=self.model, messages=messages, stream=False)
                return response['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"
    
    def stream_chat(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """Stream chat response token by token."""
        try:
            for chunk in self.client.chat(model=self.model, messages=messages, stream=True):
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def chat_with_history(self, user_message: str) -> str:
        """Chat with automatic conversation history management."""
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Get response
        response = self.chat(self.conversation_history)
        
        # Add assistant response to history
        if not response.startswith("Error:"):
            self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def stream_chat_with_history(self, user_message: str) -> Generator[str, None, None]:
        """Stream chat with automatic conversation history management."""
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Stream response and collect full response
        full_response = ""
        for chunk in self.stream_chat(self.conversation_history):
            if not chunk.startswith("Error:"):
                full_response += chunk
            yield chunk
        
        # Add complete assistant response to history
        if full_response and not full_response.startswith("Error:"):
            self.conversation_history.append({"role": "assistant", "content": full_response})
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []


# Simple test
if __name__ == "__main__":
    print("🚀 Testing Simple Ollama Client")
    print("=" * 40)
    
    client = OllamaClient()
    
    # Test connection
    if client.check_connection():
        print("✅ Connected to Ollama server")
        
        # List models
        models = client.list_models()
        print(f"📋 Available models: {models}")
        
        # Simple generation test
        print("\n🗣️ Testing generation...")
        response = client.generate("Say hello in one sentence")
        print(f"🤖 Response: {response}")
        
        # Test chat with history
        print("\n💬 Testing chat with history...")
        client.clear_conversation()
        
        response1 = client.chat_with_history("Hi, my name is Alice!")
        print(f"User: Hi, my name is Alice!")
        print(f"Bot: {response1}")
        
        response2 = client.chat_with_history("What's my name?")
        print(f"User: What's my name?")
        print(f"Bot: {response2}")
        
        print(f"\n📊 Conversation length: {len(client.conversation_history)} messages")
        
    else:
        print("❌ Cannot connect to Ollama server")
        print("💡 Start Ollama with: ollama serve")
        print("💡 Pull model with: ollama pull gemma3:1b")
