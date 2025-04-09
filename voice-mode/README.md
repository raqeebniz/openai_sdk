# Voice Mode Agent (openai_sdk)

A voice-enabled agent application that allows users to interact with an AI assistant through voice commands. This is part of the openai_sdk project.

## Features

- Voice input/output capabilities
- Real-time audio processing
- Interactive TUI (Text User Interface)
- Support for multiple languages
- Weather information tool

## Prerequisites

- Python 3.11 or higher
- Sound device (microphone and speakers)
- OpenAI API key

## Installation

1. Make sure you're in the voice-mode directory:
```bash
cd voice-mode
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -e .
```

4. Create a `.env` file in the voice-mode directory and add your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

Run the application:
```bash
python main.py
```

### Controls
- Press `K` to start/stop recording
- Press `Q` to quit the application

## Project Structure

- `main.py`: Main application entry point
- `agent1.py`: Agent workflow implementation
- `pyproject.toml`: Project dependencies and metadata
- `.env.example`: Template for environment variables
- `.chainlit/`: Chainlit configuration directory

## Dependencies

The project uses the following main dependencies:
- chainlit>=2.4.302
- openai-agents[voice]>=0.0.7
- sounddevice>=0.5.1
- whisper>=1.1.10

## License

[Your chosen license]

## Contributing

[Your contribution guidelines]
