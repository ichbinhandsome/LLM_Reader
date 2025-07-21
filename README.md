# 🦙 LLM Reader

A web-based chat interface for local Ollama models with PDF upload and analysis capabilities.

## ✨ Features

- 💬 **Real-time Chat** - Stream responses from local Ollama models
- 📄 **PDF Analysis** - Upload PDFs and ask questions about their content
- 🔄 **Clean Interface** - Simple conversation management and reset
- 🚀 **Fast** - Uses `gemma3:1b` model for quick responses

## 🚀 Quick Start

### Prerequisites
1. Install [uv](https://docs.astral.sh/uv/) - Fast Python package manager
2. Install [Ollama](https://ollama.ai)
3. Start Ollama: `ollama serve`
4. Pull model: `ollama pull gemma3:1b`  or `gemma3:4b`

### Installation & Run
```bash
cd LLM_Reader
# Run with default model (gemma3:4b)
uv run gradio_chat.py
# Run with specific model
uv run gradio_chat.py --model gemma3:1b
```

Open `http://localhost:7860` in your browser (or your custom host:port).

## 🎯 Usage

- **Chat**: Type messages and get streaming responses
- **PDF Upload**: Click "Upload PDF" to analyze documents
- **Reset**: Clear conversation and PDF content
- **Model Selection**: Use `--model` parameter to specify which Ollama model to use
- **Auto Cleanup**: Model is automatically unloaded when stopping the interface (Ctrl+C)

## 🔧 Configuration

### Command Line Options
```bash
uv run gradio_chat.py --help
```

Available options:
- `--model MODEL_NAME` - Specify Ollama model (default: gemma3:4b)
- `--host HOST` - Set host address (default: 127.0.0.1)
- `--port PORT` - Set port number (default: 7860)


## 📦 Dependencies

- `gradio>=4.0.0` - Web interface framework
- `ollama>=0.3.0` - Ollama Python client
- `pymupdf4llm>=0.0.5` - PDF processing and markdown conversion

## 📄 License

This project is open source and available under the [MIT License](LICENSE).