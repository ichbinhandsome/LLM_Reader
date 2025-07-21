"""
Gradio web chat interface for Ollama.
"""
import gradio as gr
import time
import argparse
import signal
import sys
import atexit
from typing import List, Tuple, Generator
from ollama_client import OllamaClient
import logging
import pymupdf4llm
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for cleanup
global_chat_interface = None
cleanup_called = False


def cleanup_handler():
    """Cleanup function called on exit to stop the model."""
    global global_chat_interface, cleanup_called
    
    # Prevent multiple cleanup calls
    if cleanup_called:
        return
    cleanup_called = True
    
    if global_chat_interface and global_chat_interface.client:
        logger.info("🧹 Cleaning up - stopping model...")
        global_chat_interface.client.cleanup()


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    logger.info(f"\n🛑 Received signal {signum}. Stopping model and shutting down...")
    cleanup_handler()
    sys.exit(0)


class ChatInterface:
    """Gradio chat interface for Ollama."""
    
    def __init__(self, model: str = "gemma3:4b"):
        self.client = OllamaClient(model=model)
        self.pdf_content = ""  # Store PDF content for history display
        self.conversation_history_text = ""  # Store conversation for display
        
    def check_ollama_status(self) -> Tuple[str, str]:
        """Check if Ollama is running and return status."""
        if self.client.check_connection():
            models = self.client.list_models()
            return "🟢 Connected", f"Available models: {', '.join(models)}"
        else:
            return "🔴 Disconnected", "Ollama server is not running. Please start it with: ollama serve"
    
    def update_model(self, model_name: str) -> str:
        """Update the model being used."""
        self.client.model = model_name
        self.client.clear_conversation()  # Reset conversation when changing models
        return f"Model updated to: {model_name}"
    
    def reset_conversation(self) -> Tuple[List, str, str]:
        """Reset the conversation history and start fresh."""
        self.client.clear_conversation()
        self.pdf_content = ""
        self.conversation_history_text = ""
        return [], "✅ New conversation started!", "No PDF uploaded yet."
    
    def upload_pdf(self, file) -> str:
        """Handle PDF file upload and extract text."""
        if file is None:
            return "No file uploaded"
        
        try:
            # Extract text from PDF
            with open(file, 'rb') as pdf_file:
                md_text  = pymupdf4llm.to_markdown(pdf_file)
            
            # Add the PDF content to conversation
            if md_text.strip():
                self.pdf_content = md_text  # Store for history display
                self.client.chat_with_history(f"I've uploaded a PDF. Here's the content: {md_text}")
                return f"✅ PDF processed! {len(md_text.splitlines())} pages, {len(md_text.split(' '))} words extracted. You can now ask questions about the content."
            else:
                return "❌ No text could be extracted from the PDF"
                
        except Exception as e:
            return f"❌ Error processing PDF: {str(e)}"
    
    def get_pdf_content_display(self) -> str:
        """Return the PDF content for display."""
        if not self.pdf_content:
            return "No PDF uploaded yet."
        return self.pdf_content  # Return full content without truncation
    
    def chat_response(self, message: str, history: List[List[str]]) -> Generator[Tuple[List[List[str]], str], None, None]:
        """Generate chat response with streaming using ollama_client history functions."""
        if not message.strip():
            yield history, ""
            return
            
        # Add user message to Gradio history
        history.append([message, ""])
        
        # Stream the response using ollama_client's history management
        assistant_response = ""
        try:
            for chunk in self.client.stream_chat_with_history(message):
                if not chunk.startswith("Error:"):
                    assistant_response += chunk
                    history[-1][1] = assistant_response
                    yield history, ""
                    time.sleep(0.01)  # Small delay for smooth streaming
                else:
                    # Handle error
                    history[-1][1] = chunk
                    yield history, ""
                    break
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history[-1][1] = error_msg
            yield history, ""
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(
            title="Ollama Chat Interface",
            theme=gr.themes.Default(),
            css="""
            .chatbot { height: 500px !important; }
            .chat-message { padding: 10px; margin: 5px 0; border-radius: 10px; }
            """
        ) as interface:
            
            gr.Markdown("# 🦙 Ollama Chat Interface")
            gr.Markdown("Chat with your local Ollama models through a web interface.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Simple controls
                    with gr.Group():
                        gr.Markdown("### Controls")
                        
                        # Current model display
                        current_model = gr.Textbox(
                            label="Current Model", 
                            value=self.client.model,
                            interactive=False
                        )
                        
                        # PDF upload
                        pdf_upload = gr.File(
                            label="📄 Upload PDF",
                            file_types=[".pdf"],
                            type="filepath"
                        )
                        pdf_status = gr.Textbox(label="PDF Status", interactive=False)
                        
                        # Reset conversation
                        reset_btn = gr.Button("Reset Conversation", variant="stop")
                        reset_status = gr.Textbox(label="Reset Status", interactive=False)
                
                with gr.Column(scale=3):
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="Chat",
                        height=500,
                        show_label=True,
                        container=True,
                        bubble_full_width=False
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Message",
                            placeholder="Type your message here...",
                            lines=2,
                            scale=4
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
            
            # Add usage instructions
            with gr.Accordion("📖 Usage Instructions", open=False):
                gr.Markdown("""
                ### How to use:
                1. **Start Ollama**: Make sure Ollama is running (`ollama serve`)
                2. **Start Chatting**: Type your message and press Enter or click Send
                3. **Upload PDF**: Use the PDF upload feature to analyze documents
                4. **Reset**: Use "Reset Conversation" to start fresh
                
                ### Tips:
                - Responses are streamed in real-time
                - Conversation history is managed automatically
                - PDF content can be discussed after upload
                - Use clear, specific prompts for best results
                """)
            
            # PDF content section
            with gr.Accordion("Uploaded PDF Content", open=False):
                pdf_content_display = gr.Markdown(
                    value="No PDF uploaded yet.",
                    height=400
                )
            
            # Event handlers
            def reset_handler():
                chatbot_clear, status, pdf_clear = self.reset_conversation()
                return chatbot_clear, status, pdf_clear
            
            def pdf_upload_handler(file):
                status = self.upload_pdf(file)
                return status, self.get_pdf_content_display()
            
            # Bind events
            reset_btn.click(
                reset_handler,
                outputs=[chatbot, reset_status, pdf_content_display]
            )
            
            pdf_upload.upload(
                pdf_upload_handler,
                inputs=[pdf_upload],
                outputs=[pdf_status, pdf_content_display]
            )
            
            # Chat functionality
            msg.submit(
                self.chat_response,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            send_btn.click(
                self.chat_response,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
        
        return interface


def main():
    """Main function to run the chat interface."""
    global global_chat_interface
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ollama Chat Interface with Gradio")
    parser.add_argument(
        "--model", 
        type=str, 
        default="gemma3:4b",
        help="Ollama model to use for chat (default: gemma3:4b)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to run the Gradio interface on (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio interface on (default: 7860)"
    )
    
    args = parser.parse_args()
    
    print(f"🤖 Starting Ollama Chat Interface with model: {args.model}")
    
    # Initialize chat interface with specified model
    chat_interface = ChatInterface(model=args.model)
    global_chat_interface = chat_interface  # Store for cleanup
    
    # Register cleanup handlers
    atexit.register(cleanup_handler)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check if Ollama is running
    status, details = chat_interface.check_ollama_status()
    print(f"Ollama Status: {status}")
    print(f"Details: {details}")
    
    if "Disconnected" in status:
        print("\n⚠️  Warning: Ollama server not detected!")
        print("Please start Ollama server first:")
        print("  ollama serve")
        print("\nThe interface will still launch, but you'll need to start Ollama to chat.")
    
    # Create and launch interface
    interface = chat_interface.create_interface()
    
    print(f"\n🚀 Launching Gradio interface on {args.host}:{args.port}...")
    print(f"💬 Using model: {args.model}")
    print("💬 Open your browser and start chatting with Ollama!")
    print("💡 Press Ctrl+C to stop the interface and unload the model")
    
    try:
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=False,
            debug=False,
            show_error=True,
            inbrowser=True            # Automatically open browser
        )
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received")
        # cleanup_handler() will be called by signal handler or atexit
    except Exception as e:
        print(f"\n❌ Error running interface: {e}")
        cleanup_handler()
    finally:
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()
