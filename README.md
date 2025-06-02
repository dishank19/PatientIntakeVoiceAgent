# Neurality Health AI Voice Agent

A voice-based AI assistant for healthcare patient engagement, capable of understanding and responding to patient requests in multiple languages.

## Features

- Real-time speech-to-text transcription
- Multi-language support (English and Spanish)
- Intent classification for patient requests
- Natural language response generation
- Structured data storage in JSON format
- Integration with Daily.co for voice calls
- Cartesia TTS for natural voice responses

## Prerequisites

- Python 3.9+
- OpenAI API key
- Daily.co API key
- Cartesia API key (optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/pipecat-voice-agent.git
cd pipecat-voice-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the environment template and fill in your API keys:
```bash
cp .env.example .env
```

## Configuration

Edit the `.env` file with your API keys and configuration:

```env
OPENAI_API_KEY=your_openai_api_key_here
DAILY_API_KEY=your_daily_api_key_here
CARTESIA_API_KEY=your_cartesia_api_key_here
DAILY_ROOM_URL=https://your-domain.daily.co/your-room
DAILY_ROOM_NAME=your-room-name
```

## Usage

1. Start the voice agent:
```bash
python src/voice_agent/main.py
```

2. Join the Daily.co room using the provided URL.

3. The agent will:
   - Transcribe your speech
   - Classify your intent
   - Generate appropriate responses
   - Store the conversation data

## Project Structure

```
pipecat-voice-agent/
├── src/
│   └── voice_agent/
│       ├── core/           # Core functionality
│       ├── services/       # External service integrations
│       ├── utils/          # Utility functions
│       └── config/         # Configuration management
├── tests/                  # Test suite
├── data/                   # Data storage
├── assets/                 # Static assets
├── .env.example           # Environment template
├── requirements.txt       # Dependencies
└── README.md             # Documentation
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
ruff check src/ tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Security

- API keys are stored in environment variables
- All data is stored locally
- No sensitive patient information is transmitted to third parties
- Regular security audits are performed

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
