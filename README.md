# patient — Voice Agent

## Overview
A production-ready, multimodal voice agent for patient engagement, built with FastAPI, Pipecat, and modern LLMs. Supports both local and Dockerized workflows.

---

## 1. Environment Setup

### 1.1. Clone the Repository
```sh
git clone <your-repo-url>
cd pipecat-voice-agent
```

### 1.2. Prepare the Environment File
- Copy the example file and fill in your API keys:
```sh
cp .env.example .env
# Edit .env and add your credentials:
# DAILY_API_KEY=...
# DAILY_API_URL=https://api.daily.co/v1
# TWILIO_ACCOUNT_SID=...
# TWILIO_AUTH_TOKEN=...
# OPENAI_API_KEY=...
# CARTESIA_API_KEY=...
```

---

## 2. Running with Docker (Recommended)

### 2.1. Build the Docker Image
```sh
docker-compose build
```

### 2.2. Start the Service
```sh
docker-compose up
```

- The app will be available at [http://localhost:7860/](http://localhost:7860/)
- The `.env` file is mounted at runtime (not baked into the image).
- Data and logs are persisted in the `./data` directory.

---

## 3. Running Locally (Without Docker)

### 3.1. Create a Virtual Environment
```sh
python3 -m venv venv
source venv/bin/activate
```

### 3.2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3.3. Run the Server
```sh
python server.py
```
- The app will be available at [http://localhost:7860/](http://localhost:7860/)

---

## 4. Running with an Audio File (Batch Mode)

You can process a `.wav` or `.mp3` file directly:
```sh
python server.py -au /path/to/yourfile.wav
```
- The agent's response will be printed to the console.

---

## 5. Twilio & Webhook Setup

### 5.1. Install and Start ngrok (for Local Testing)

1. **Install ngrok:**
   - [Download ngrok](https://ngrok.com/download) and follow the install instructions for your OS.
   - Or, on macOS:
     ```sh
     brew install ngrok
     ```
2. **Authenticate ngrok (first time only):**
   - Sign up at [ngrok.com](https://ngrok.com/) and get your authtoken from your dashboard.
   - Run:
     ```sh
     ngrok config add-authtoken <your-ngrok-authtoken>
     ```
3. **Start ngrok on your FastAPI port (default 7860):**
   ```sh
   ngrok http 7860
   ```
4. **Copy the HTTPS Forwarding URL** shown in the ngrok terminal (e.g., `https://xxxx-xx-xx-xx.ngrok-free.app`).

### 5.2. Configure Twilio Webhooks

1. Go to your [Twilio Console](https://console.twilio.com/).
2. Navigate to **Phone Numbers > Manage > Active numbers** and select your number.
3. Under **Voice & Fax** configuration:
   - **A call comes in:**
     - Set to **Webhook**
     - URL: `https://<your-ngrok-url>/call` (e.g., `https://xxxx-xx-xx-xx.ngrok-free.app/call`)
     - HTTP POST
   - **Call status changes:**
     - URL: `https://<your-ngrok-url>/call-status`
     - HTTP POST
4. **Save** your changes.

### 5.3. Test the Integration
- Call your Twilio number. You should hear hold music, then be connected to your voice agent.
- The ngrok terminal will show incoming requests.

---

## 6. .env.example
```
DAILY_API_KEY=your_daily_api_key
DAILY_API_URL=https://api.daily.co/v1
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
OPENAI_API_KEY=your_openai_api_key
CARTESIA_API_KEY=your_cartesia_api_key
```

---

## 7. Troubleshooting
- Ensure your `.env` file is present and filled out.
- For Docker, use `docker-compose logs` to view logs.
- For local, check the terminal output for errors.
- If you change dependencies, rebuild the Docker image.

