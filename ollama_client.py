"""
Simple Ollama API client using ollama-python library.
"""
import ollama
from typing import Dict, List, Generator, Union, Optional
import logging
import subprocess
import atexit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """Simple client for interacting with Ollama."""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = "gemma3:4b"):
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
    
    def chat_with_history(self, user_message: str, image_path: Optional[str] = None) -> str:
        """Chat with automatic conversation history management, optionally with image."""
        # Prepare message for history
        message = {"role": "user", "content": user_message}
        if image_path:
            message["images"] = [image_path]
        
        # Add user message to history
        self.conversation_history.append(message)
        
        # Get response using direct ollama chat
        try:
            if image_path:
                # Use ollama.chat directly for image messages
                response = self.client.chat(
                    model=self.model,
                    messages=self.conversation_history,
                    stream=False
                )
                response_content = response['message']['content']
            else:
                # Use regular chat method for text-only
                response_content = self.chat(self.conversation_history)
        except Exception as e:
            return f"Error: {str(e)}"
        
        # Add assistant response to history
        if not response_content.startswith("Error:"):
            self.conversation_history.append({"role": "assistant", "content": response_content})
        
        return response_content
    
    def stream_chat_with_history(self, user_message: str, image_path: Optional[str] = None) -> Generator[str, None, None]:
        """Stream chat with automatic conversation history management, optionally with image."""
        # Prepare message for history
        message = {"role": "user", "content": user_message}
        if image_path:
            message["images"] = [image_path]
        
        # Add user message to history
        self.conversation_history.append(message)
        
        # Stream response and collect full response
        full_response = ""
        try:
            if image_path:
                # Use ollama client directly for streaming with images
                for chunk in self.client.chat(model=self.model, messages=self.conversation_history, stream=True):
                    if 'message' in chunk and 'content' in chunk['message']:
                        chunk_content = chunk['message']['content']
                        full_response += chunk_content
                        yield chunk_content
            else:
                # Use regular stream chat method for text-only
                for chunk in self.stream_chat(self.conversation_history):
                    if not chunk.startswith("Error:"):
                        full_response += chunk
                    yield chunk
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            yield error_msg
            return
        
        # Add complete assistant response to history
        if full_response and not full_response.startswith("Error:"):
            self.conversation_history.append({"role": "assistant", "content": full_response})
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def stop_model(self, model_name: str = None) -> bool:
        """Stop/unload a specific model from memory."""
        target_model = model_name or self.model
        try:
            logger.info(f"🛑 Stopping model: {target_model}")
            # Use subprocess to call ollama stop command
            result = subprocess.run(
                ["ollama", "stop", target_model],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Model {target_model} stopped successfully")
                return True
            else:
                logger.warning(f"⚠️ Model {target_model} may not have been loaded or already stopped")
                return True  # Not necessarily an error
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Timeout stopping model {target_model}")
            return False
        except FileNotFoundError:
            logger.error("❌ Ollama command not found. Please install Ollama first.")
            return False
        except Exception as e:
            logger.error(f"❌ Error stopping model {target_model}: {e}")
            return False
    
    def cleanup(self):
        """Cleanup function to stop the current model."""
        if self.model:
            self.stop_model(self.model)


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
        
        # Test image handling (if images available)
        print("\n🖼️ Testing image handling...")
        try:
            # This will fail if no image is available, but shows the API usage
            response3 = client.chat_with_history("What's in this image?", "test.jpg")
            print(f"User: What's in this image? [with image: test.jpg]")
            print(f"Bot: {response3}")
        except Exception as e:
            print(f"Note: Image test skipped (no test image available): {e}")
        
    else:
        print("❌ Cannot connect to Ollama server")
        print("💡 Start Ollama with: ollama serve")
        print("💡 Pull model with: ollama pull gemma3:1b")
