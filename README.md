# Patient Interaction Voice Agent

## Overview
A voice agent for patient engagement, designed to handle incoming calls, collect patient information, and schedule appointments. It integrates with Twilio for telephony, uses Daily.co for managing WebRTC sessions for the bot, OpenAI for natural language understanding and function calling, and Cartesia for text-to-speech (TTS) synthesis. The backend is built with FastAPI, and the system supports both local execution and Dockerized workflows.

## Key Features
- **Voice Interaction**: Engages users in natural conversation via phone calls (Twilio integration).
- **Intent Classification**: Determines patient intent (e.g., scheduling, information query).
- **Information Gathering**: Collects patient name, reason for visit, and medical details (allergies, prescriptions, conditions).
- **Appointment Scheduling**: Checks for available slots, suggests alternatives, and confirms bookings.
- **Persistent Appointment Storage**: Stores confirmed appointments in a JSON file (`data/appointments.json`).
- **Call Logging**: Saves detailed logs for each call interaction in the `data/` directory.
- **Web Dashboard**: Provides a simple UI to view scheduled appointments, filterable by date.
- **Bilingual Support**: Basic support for English and Spanish interactions.

## Technical Stack
- **Backend Framework**: FastAPI
- **Telephony**: Twilio (for receiving calls)
- **Bot Transport & WebRTC**: Daily.co (via Pipecat-AI)
- **Natural Language Understanding (NLU) / LLM**: OpenAI (GPT models)
- **Text-to-Speech (TTS)**: Cartesia
- **Core AI/Voice Framework**: Pipecat-AI
- **Programming Language**: Python
- **Data Storage**: JSON files (for appointments and call logs)
- **Containerization**: Docker, Docker Compose
- **Date/Time Parsing**: `python-dateutil`
- **HTTP Client**: `aiohttp` (used by Pipecat and for Daily REST API calls)

---

## 1. Environment Setup

### 1. Prepare the Environment File
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
4. **Copy the HTTPS Forwarding URL** shown in the ngrok terminal (e.g., `https://xxxx-xx-xx-xx.ngrok-free.app`). This URL is crucial and will be used to configure your Twilio webhooks in the next steps. Make sure to copy the `https` URL.

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

---

## 8. Assumptions and Simplifications
This project, while functional, includes several simplifications for development purposes:
- **Data Storage**: Uses JSON files (`data/appointments.json`, `data/call_log_*.json`) for simplicity. A production system would typically use a robust database (e.g., PostgreSQL, MySQL) with proper indexing, transactions, and backup strategies.
- **Concurrency Handling**: Basic file locking (`fcntl`) is used for `appointments.json`. High-concurrency production environments would require more sophisticated distributed locking or a database that handles concurrency.
- **Error Handling**: Error handling is present but could be made more comprehensive for production scenarios (e.g., more specific error types, retry mechanisms for external API calls).
- **Security**: 
    - The dashboard (`/dashboard/appointments`) is currently open. Production would require authentication and authorization.
    - Twilio webhook validation is present but relies on `TWILIO_AUTH_TOKEN` being set; robust secret management is crucial.
    - API keys are managed via `.env` files. Production systems need secure secret management (e.g., HashiCorp Vault, AWS Secrets Manager, Azure Key Vault).
- **Scalability**: The current setup (single server process, file-based storage) is suitable for single-instance testing. Production deployment would require a scalable architecture (e.g., multiple server instances behind a load balancer, distributed task queues if bot processes become long-running or resource-intensive, a scalable database).
- **Date/Time Zone Handling**: While `dateutil` helps with parsing, rigorous time zone management across all components (user input, server, database, display) is critical in production and is currently simplified.
- **Mock Data**: The `GET /` endpoint initiates a bot with a mock CallSid for easy testing. This should be disabled or protected in a production environment.
- **Business Hours**: Hardcoded in `src/scheduling_helpers.py`; these should ideally be configurable.
- **Twilio Call Forwarding**: The bot currently answers the call and then updates Twilio to forward to a Daily SIP URI. More advanced scenarios might involve different call handling strategies.

---

## 9. Production Considerations for Healthcare
Deploying a patient interaction system in a healthcare context requires careful attention to several critical areas beyond the scope of this example project:

- **HIPAA/PIPEDA/GDPR Compliance**: 
    - All components handling Protected Health Information (PHI) or Personally Identifiable Information (PII) must be compliant with relevant data privacy and security regulations (e.g., HIPAA in the US, PIPEDA in Canada, GDPR in Europe).
    - This includes infrastructure, data storage, transmission (encryption in transit and at rest), access controls, audit logging, and Business Associate Agreements (BAAs) with all third-party vendors (Twilio, Daily, OpenAI, Cartesia, hosting provider).
- **Data Security & Encryption**: 
    - **Encryption at Rest**: PHI stored (e.g., in `appointments.json` or a database) must be encrypted.
    - **Encryption in Transit**: All communication channels (user to Twilio, Twilio to server, server to Daily/OpenAI/Cartesia, dashboard access) must use strong encryption (e.g., HTTPS/TLS, WSS).
    - **Access Controls**: Role-based access control (RBAC) for any administrative interfaces (like the dashboard) or direct data access.
    - **Audit Trails**: Comprehensive logging of access to PHI and significant system events.
- **Vendor Selection**: Ensure all third-party services (telephony, STT/TTS, LLM, hosting) are HIPAA-compliant (or meet equivalent local regulations) and that BAAs are in place.
    - **OpenAI**: Review OpenAI's policies regarding PHI and BAAs. As of early 2024, using OpenAI for PHI typically requires their enterprise offerings or specific agreements.
    - **Cartesia, Daily, Twilio**: Verify their compliance statements and BAA availability for healthcare use cases.
- **Reliability and Availability**: 
    - High availability architecture (e.g., redundant servers, load balancing, database replication).
    - Robust monitoring and alerting for system health and performance.
    - Disaster recovery and business continuity plans.
- **Scalability**: Design to handle peak call volumes and data storage growth.
- **Error Handling and Fallbacks**: Graceful degradation if a service (e.g., TTS, LLM) fails. Clear error communication to users and staff.
- **User Authentication and Authorization**: Secure access to the dashboard and any administrative functions.
- **Logging and Monitoring**: Detailed logging for operational monitoring, debugging, and security auditing, ensuring logs themselves do not inappropriately expose PHI unless properly secured.
- **Accuracy and Safety**: 
    - Rigorous testing of the LLM's understanding and responses, especially concerning medical information or scheduling accuracy.
    - Clear disclaimers that the agent is not a medical professional and cannot provide medical advice.
    - Mechanisms for human oversight or intervention if the bot encounters situations it cannot handle or makes errors.
- **Call Recording Policies**: If calls are recorded (not explicitly implemented here but common), ensure compliance with consent laws (e.g., two-party consent where required) and secure storage/access for recordings containing PHI.
- **Deployment Environment**: Secure and compliant hosting environment (e.g., HIPAA-eligible services on AWS, Azure, GCP, or a compliant private cloud).

---

## 10. Troubleshooting
- Ensure your `.env` file is present and filled out.
- For Docker, use `docker-compose logs` to view logs.
- For local, check the terminal output for errors.
- If you change dependencies, rebuild the Docker image.

---

## 11. Testing and Logging

### 11.1. Running Tests
To run the tests, execute the following command:
```sh
python -m pytest
```

### 11.2. Logging
Logs are stored in the `./data` directory. You can view them using:
```sh
cat ./data/logs/app.log
```

**Disclaimer:** The testing and logging functionality may not work as intended. Please report any issues you encounter.

