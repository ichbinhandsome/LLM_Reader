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
        self.current_image_path = None  # Store current uploaded image path
        self.url_content = ""  # Store URL content for history display
        self.jina_api_key = ""  # Store Jina API key if needed

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
    
    def reset_conversation(self) -> Tuple[List, str, str, str, str]:
        """Reset the conversation history and start fresh."""
        self.client.clear_conversation()
        self.pdf_content = ""
        self.conversation_history_text = ""
        self.current_image_path = None
        self.url_content = ""
        return [], "✅ New conversation started!", "No PDF uploaded yet.", "No image uploaded yet.", "No URL processed yet."
    
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
    
    def upload_image(self, file) -> str:
        """Handle image file upload."""
        if file is None:
            self.current_image_path = None
            return "No image uploaded"
        
        try:
            # Store the image path for later use
            self.current_image_path = file
            
            # Get basic info about the image
            file_size = os.path.getsize(file) / 1024  # KB
            file_name = os.path.basename(file)
            
            return f"✅ Image uploaded: {file_name} ({file_size:.1f} KB). You can now ask questions about this image."
                
        except Exception as e:
            self.current_image_path = None
            return f"❌ Error processing image: {str(e)}"
    
    def process_url(self, url: str, api_key: str = "") -> str:
        """Handle website URL processing using Jina Reader API."""
        if not url or not url.strip():
            return "No URL provided"

        url = url.strip()
        
        # Basic URL validation
        if not url.startswith(('http://', 'https://')):
            return "❌ Invalid URL. Please provide a valid URL starting with http:// or https://"
        
        try:
            import requests
            
            # Store the API key if provided
            if api_key and api_key.strip():
                self.jina_api_key = api_key.strip()
            
            # Construct Jina Reader API URL
            jina_url = "https://r.jina.ai/" + url
            
            # Set up headers
            headers = {
                "Accept": "application/json",
                "X-No-Cache": "true",
                "X-Retain-Images": "none",
                "X-Timeout": "10"
            }
            
            # Add authorization if API key is available
            if self.jina_api_key:
                headers["Authorization"] = f"Bearer {self.jina_api_key}"
            
            # Make the request
            response = requests.get(jina_url, headers=headers)
            
            if response.status_code == 200:
                # Parse the JSON response
                data = response.json()
                
                # Extract content from the response
                if 'data' in data and 'content' in data['data']:
                    content = data['data']['content']
                    self.url_content = content
                    
                    # Add the URL content to conversation history
                    self.client.chat_with_history(f"I've processed a website URL: {url}. Here's the content: {content}")
                    
                    # Calculate some stats
                    word_count = len(content.split())
                    char_count = len(content)
                    
                    return f"✅ URL processed successfully!\nURL: {url}\nExtracted {word_count} words ({char_count} characters). You can now ask questions about the content."
                else:
                    return f"❌ No content found in the response from: {url}"
            else:
                return f"❌ Failed to process URL. Status code: {response.status_code}\nError: {response.text}"
                
        except requests.RequestException as e:
            return f"❌ Network error processing URL: {str(e)}"
        except Exception as e:
            return f"❌ Error processing URL: {str(e)}"
    
    def get_image_display(self) -> str:
        """Return the current image path for display."""
        if not self.current_image_path:
            return None
        return self.current_image_path
    
    def get_pdf_content_display(self) -> str:
        """Return the PDF content for display."""
        if not self.pdf_content:
            return "No PDF uploaded yet."
        return self.pdf_content  # Return full content without truncation
    
    def get_url_content_display(self) -> str:
        """Return the URL content for display."""
        if not self.url_content:
            return "No URL processed yet."
        return self.url_content
    
    def chat_response(self, message: str, history: List[List[str]]) -> Generator[Tuple[List[List[str]], str], None, None]:
        """Generate chat response with streaming using ollama_client history functions."""
        if not message.strip() and not self.current_image_path:
            yield history, ""
            return
            
        # Add user message to Gradio history with image reference
        if self.current_image_path:
            # Display message with image reference (image shown in sidebar)
            if message.strip():
                display_text = f"📷 {message}\n\n[Image uploaded: {os.path.basename(self.current_image_path)}]"
            else:
                display_text = f"📷 Please analyze the uploaded image\n\n[Image: {os.path.basename(self.current_image_path)}]"
            history.append([display_text, ""])
        else:
            history.append([message, ""])
        
        # Stream the response using ollama_client's history management with optional image
        assistant_response = ""
        try:
            for chunk in self.client.stream_chat_with_history(message, self.current_image_path):
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
            
            # Clear the image after processing to avoid reusing it
            self.current_image_path = None
            
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
            gr.Markdown("Chat with your local Ollama models through a web interface. **Upload images** for vision model analysis!")
            
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
                        
                        # Upload type selection
                        upload_type = gr.CheckboxGroup(
                            label="📂 Select Upload Types",
                            choices=["PDF Document", "Image", "Website URL"],
                            value=[],
                            interactive=True
                        )
                        
                        # PDF upload section (initially hidden)
                        with gr.Row(visible=False) as pdf_section:
                            with gr.Column():
                                pdf_upload = gr.File(
                                    label="📄 Upload PDF",
                                    file_types=[".pdf"],
                                    type="filepath"
                                )
                                pdf_status = gr.Textbox(label="PDF Status", interactive=False)
                                
                                # PDF content display
                                with gr.Accordion("📄 Uploaded PDF Content", open=False):
                                    pdf_content_display = gr.Markdown(
                                        value="No PDF uploaded yet.",
                                        height=400
                                    )
                        
                        # Image upload section (initially hidden)
                        with gr.Row(visible=False) as image_section:
                            with gr.Column():
                                image_upload = gr.File(
                                    label="🖼️ Upload Image",
                                    file_types=[".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"],
                                    type="filepath"
                                )
                                image_status = gr.Textbox(label="Image Status", interactive=False)
                                
                                # Current uploaded image display (bigger size)
                                current_image = gr.Image(
                                    label="📷 Current Image",
                                    height=250,
                                    show_label=True,
                                    visible=True,
                                    interactive=False
                                )
                        
                        # Website URL upload section (initially hidden)
                        with gr.Row(visible=False) as url_section:
                            with gr.Column():
                                url_input = gr.Textbox(
                                    label="🌐 Enter Website URL",
                                    placeholder="https://example.com",
                                    lines=1,
                                    interactive=True
                                )
                                jina_api_key = gr.Textbox(
                                    label="Jina API Key (Optional)",
                                    placeholder="Enter your Jina API Key for better rate limits",
                                    lines=1,
                                    interactive=True,
                                    type="password"
                                )
                                url_submit = gr.Button("📥 Process URL", variant="secondary")
                                url_status = gr.Textbox(label="URL Status", interactive=False)
                                
                                # URL content display
                                with gr.Accordion("🌐 Website URL Content", open=False):
                                    url_content_display = gr.Markdown(
                                        value="No URL processed yet.",
                                        height=400
                                    )
                
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
                        reset_btn = gr.Button("🔄 Reset Chat", variant="stop", scale=1)
                    
                    with gr.Row():
                        send_btn = gr.Button("Send", variant="primary", scale=1)
            
            # Add usage instructions
            with gr.Accordion("📖 Usage Instructions", open=False):
                gr.Markdown("""
                ### How to use:
                1. **Start Ollama**: Make sure Ollama is running (`ollama serve`)
                2. **Select Upload Types**: Check "PDF Document", "Image", and/or "Website URL" to show upload options
                3. **Upload Files/URLs**: Use the upload boxes or URL input that appear based on your selection
                4. **For URLs**: Optionally provide a Jina API key for better rate limits and performance
                5. **Start Chatting**: Type your message and press Enter or click Send
                6. **Reset**: Use "Reset Conversation" to start fresh
                
                ### Tips:
                - You can upload PDF, Image, and process URLs simultaneously
                - Responses are streamed in real-time
                - Conversation history is managed automatically
                - PDF content can be discussed after upload
                - Images are processed once per upload - upload again for new image analysis
                - Website URLs are processed using Jina Reader API (works without API key but with rate limits)
                - Jina API key is optional but recommended for better performance and higher rate limits
                - For best image analysis, use vision models like `llava:7b` or `llava:13b`
                - Use clear, specific prompts for best results
                """)
            
            # Event handlers
            def toggle_upload_sections(selected_types):
                """Show/hide upload sections based on selection."""
                show_pdf = "PDF Document" in selected_types
                show_image = "Image" in selected_types
                show_url = "Website URL" in selected_types
                return {
                    pdf_section: gr.update(visible=show_pdf),
                    image_section: gr.update(visible=show_image),
                    url_section: gr.update(visible=show_url)
                }
            
            def reset_handler():
                chatbot_clear, status, pdf_clear, image_clear, url_clear = self.reset_conversation()
                # Show a popup notification instead of updating message box
                gr.Info("✅ Chat reset successfully! Start a new conversation.")
                return chatbot_clear, "", pdf_clear, image_clear, None, url_clear, ""
            
            def pdf_upload_handler(file):
                status = self.upload_pdf(file)
                return status, self.get_pdf_content_display()
            
            def image_upload_handler(file):
                status = self.upload_image(file)
                image_display = self.get_image_display() 
                return status, image_display
            
            def url_handler(url, api_key):
                status = self.process_url(url, api_key)
                url_content = self.get_url_content_display()
                return status, "", url_content  # Clear the URL input after processing and update content display
            
            # Bind events
            upload_type.change(
                toggle_upload_sections,
                inputs=[upload_type],
                outputs=[pdf_section, image_section, url_section]
            )
            
            reset_btn.click(
                reset_handler,
                outputs=[chatbot, msg, pdf_content_display, image_status, current_image, url_content_display, url_input]
            )
            
            pdf_upload.upload(
                pdf_upload_handler,
                inputs=[pdf_upload],
                outputs=[pdf_status, pdf_content_display]
            )
            
            image_upload.upload(
                image_upload_handler,
                inputs=[image_upload],
                outputs=[image_status, current_image]
            )
            
            url_submit.click(
                url_handler,
                inputs=[url_input, jina_api_key],
                outputs=[url_status, url_input, url_content_display]
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
        help="Ollama model to use for chat (default: gemma3:4b, use llava:7b or llava:13b for image support)"
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
