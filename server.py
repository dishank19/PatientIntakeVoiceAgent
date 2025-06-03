#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os
import subprocess
from contextlib import asynccontextmanager

import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Optional, List
import json
import asyncio
from datetime import datetime
import wave

from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomParams, DailyRoomProperties, DailyRoomSipParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMContext, OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from twilio.twiml.voice_response import VoiceResponse, Dial, Sip
from twilio.rest import Client

# Load environment variables from .env file
load_dotenv(override=True)

MAX_BOTS_PER_ROOM = 1

# Bot sub-process dict for status reporting and concurrency control
bot_procs = {}

daily_helpers = {}

# Add these models for request/response validation
class IntentResponse(BaseModel):
    transcript: str
    language: str
    intent: str
    response: str
    confidence_score: float

class ConversationHistory(BaseModel):
    timestamp: str
    transcript: str
    intent: str
    response: str

# Add conversation history storage
conversation_history: List[ConversationHistory] = []

# Initialize global pipeline components
llm = OpenAILLMService(
    name="LLM",
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4",
)

tts = CartesiaTTSService(
    api_key=os.getenv("CARTESIA_API_KEY"),
    voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
)

# Initialize conversation context
messages = [
    {
        "role": "system",
        "content": """You are Jessica, an agent for a company called Tri-County Health Services. Your job is to collect important information from the user before their doctor visit. You should be polite and professional. You're not a medical professional, so you shouldn't provide any advice. Keep your responses short. Your job is to collect information to give to a doctor. Don't make assumptions about what values to plug into functions. Ask for clarification if a user response is ambiguous."""
    }
]

context = OpenAILLMContext(messages=messages)
context_aggregator = llm.create_context_aggregator(context)

# Initialize Twilio client
twilio_client = Client(
    os.getenv("TWILIO_ACCOUNT_SID"),
    os.getenv("TWILIO_AUTH_TOKEN")
)

def cleanup():
    # Clean up function, just to be extra safe
    for entry in bot_procs.values():
        proc = entry[0]
        proc.terminate()
        proc.wait()


@asynccontextmanager
async def lifespan(app: FastAPI):
    aiohttp_session = aiohttp.ClientSession()
    daily_helpers["rest"] = DailyRESTHelper(
        daily_api_key=os.getenv("DAILY_API_KEY", ""),
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )
    yield
    await aiohttp_session.close()
    cleanup()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def start_agent(request: Request):
    print(f"!!! Creating room")
    room = await daily_helpers["rest"].create_room(DailyRoomParams())
    print(f"!!! Room URL: {room.url}")
    # Ensure the room property is present
    if not room.url:
        raise HTTPException(
            status_code=500,
            detail="Missing 'room' property in request data. Cannot start agent without a target room!",
        )

    # Check if there is already an existing process running in this room
    num_bots_in_room = sum(
        1 for proc in bot_procs.values() if proc[1] == room.url and proc[0].poll() is None
    )
    if num_bots_in_room >= MAX_BOTS_PER_ROOM:
        raise HTTPException(status_code=500, detail=f"Max bot limited reach for room: {room.url}")

    # Get the token for the room
    token = await daily_helpers["rest"].get_token(room.url)

    if not token:
        raise HTTPException(status_code=500, detail=f"Failed to get token for room: {room.url}")

    # Spawn a new agent, and join the user session
    # Note: this is mostly for demonstration purposes (refer to 'deployment' in README)
    try:
        proc = subprocess.Popen(
            [f"python3 -m bot -u {room.url} -t {token}"],
            shell=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        bot_procs[proc.pid] = (proc, room.url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")

    return RedirectResponse(room.url)


@app.get("/status/{pid}")
def get_status(pid: int):
    # Look up the subprocess
    proc = bot_procs.get(pid)

    # If the subprocess doesn't exist, return an error
    if not proc:
        raise HTTPException(status_code=404, detail=f"Bot with process id: {pid} not found")

    # Check the status of the subprocess
    if proc[0].poll() is None:
        status = "running"
    else:
        status = "finished"

    return JSONResponse({"bot_id": pid, "status": status})


@app.post("/process-audio", response_model=IntentResponse)
async def process_audio(audio_file: UploadFile = File(...)):
    """
    Process an audio file and return intent classification and response
    """
    try:
        # Save the uploaded file temporarily
        temp_path = f"temp_{datetime.now().timestamp()}.wav"
        with open(temp_path, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # Create pipeline for processing
        pipeline = Pipeline([
            context_aggregator.user(),
            llm,
            tts,
            context_aggregator.assistant(),
        ])
        
        # Create and run pipeline task
        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            )
        )
        
        # Process the audio file
        with wave.open(temp_path, 'rb') as audio_file:
            audio_data = audio_file.readframes(-1)
            audio_frame = OutputAudioRawFrame(
                audio_data,
                audio_file.getframerate(),
                audio_file.getnchannels()
            )
            
            # Queue the audio frame for processing
            await task.queue_frames([audio_frame])
            
            # Get the response
            response = await task.get_response()
            
            # Extract intent and other information
            result = {
                "transcript": response.get("transcript", ""),
                "language": "Spanish" if response.get("is_spanish", False) else "English",
                "intent": response.get("intent", "general_inquiry"),
                "response": response.get("response", ""),
                "confidence_score": response.get("confidence_score", 0.95)
            }
            
            # Store in conversation history
            conversation_history.append(ConversationHistory(
                timestamp=datetime.now().isoformat(),
                transcript=result["transcript"],
                intent=result["intent"],
                response=result["response"]
            ))
            
            return result
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/conversation-history", response_model=List[ConversationHistory])
async def get_conversation_history():
    """
    Retrieve the conversation history
    """
    return conversation_history

@app.post("/classify-intent", response_model=IntentResponse)
async def classify_intent(text: str):
    """
    Classify the intent of a text input
    """
    try:
        # Create pipeline for processing
        pipeline = Pipeline([
            context_aggregator.user(),
            llm,
            context_aggregator.assistant(),
        ])
        
        # Create and run pipeline task
        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            )
        )
        
        # Process the text
        await task.queue_frames([text])
        response = await task.get_response()
        
        result = {
            "transcript": text,
            "language": "Spanish" if response.get("is_spanish", False) else "English",
            "intent": response.get("intent", "general_inquiry"),
            "response": response.get("response", ""),
            "confidence_score": response.get("confidence_score", 0.95)
        }
        
        # Store in conversation history
        conversation_history.append(ConversationHistory(
            timestamp=datetime.now().isoformat(),
            transcript=result["transcript"],
            intent=result["intent"],
            response=result["response"]
        ))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/call", response_class=PlainTextResponse)
async def handle_incoming_call(request: Request):
    """
    Handle incoming Twilio calls
    """
    try:
        # Get form data from Twilio
        form_data = await request.form()
        data = dict(form_data)
        call_sid = data.get('CallSid')
        
        # Create a SIP-enabled Daily room
        params = DailyRoomParams(
            properties=DailyRoomProperties(
                sip=DailyRoomSipParams(
                    display_name="health-intake-bot",
                    video=False,
                    sip_mode="dial-in",
                    num_endpoints=1
                )
            )
        )
        
        # Create room and get token
        room = await daily_helpers["rest"].create_room(params)
        token = await daily_helpers["rest"].get_token(room.url)
        
        if not room.url or not token:
            raise HTTPException(
                status_code=500,
                detail="Failed to create Daily room or get token"
            )
        
        # Start the bot process
        proc = subprocess.Popen(
            [
                f"python3 -m bot -u {room.url} -t {token} -i {call_sid} -s {room.config.sip_endpoint}"
            ],
            shell=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Store process info
        bot_procs[proc.pid] = (proc, room.url)
        
        # Put caller on hold with music while bot initializes
        resp = VoiceResponse()
        resp.play(url="http://com.twilio.sounds.music.s3.amazonaws.com/MARKOVICHAMP-Borghestral.mp3", loop=10)
        return str(resp)
        
    except Exception as e:
        logger.error(f"Error handling incoming call: {str(e)}")
        resp = VoiceResponse()
        resp.say("We're sorry, but we're experiencing technical difficulties. Please try again later.")
        return str(resp)

@app.post("/call-status")
async def handle_call_status(request: Request):
    """
    Handle Twilio call status updates
    """
    try:
        form_data = await request.form()
        data = dict(form_data)
        call_sid = data.get('CallSid')
        call_status = data.get('CallStatus')
        
        # Log call status
        logger.info(f"Call {call_sid} status: {call_status}")
        
        # If call ended, clean up resources
        if call_status == 'completed':
            # Find and terminate associated bot process
            for pid, (proc, room_url) in bot_procs.items():
                if proc.poll() is None:  # Process is still running
                    proc.terminate()
                    proc.wait()
                    del bot_procs[pid]
                    break
        
        return JSONResponse({"status": "success"})
        
    except Exception as e:
        logger.error(f"Error handling call status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    from pyngrok import ngrok
    import uvicorn
    import argparse
    import sys

    default_host = os.getenv("HOST", "0.0.0.0")
    default_port = int(os.getenv("FAST_API_PORT", "7860"))

    parser = argparse.ArgumentParser(description="Daily patient-intake FastAPI server")
    parser.add_argument("--host", type=str, default=default_host, help="Host address")
    parser.add_argument("--port", type=int, default=default_port, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Reload code on change")
    parser.add_argument("-au", "--audio-upload", type=str, help="Path to audio file for direct processing")
    config = parser.parse_args()

    if config.audio_upload:
        import requests
        import time
        # Start the server in a subprocess if not already running, or just call the endpoint directly if running
        url = f"http://localhost:{config.port}/process-audio"
        # Optionally, check if the server is running, else start it in the background
        try:
            with open(config.audio_upload, "rb") as f:
                ext = config.audio_upload.split(".")[-1].lower()
                mime = "audio/wav" if ext == "wav" else "audio/mpeg" if ext == "mp3" else "application/octet-stream"
                files = {"audio_file": (config.audio_upload, f, mime)}
                response = requests.post(url, files=files)
                print("Agent response:")
                print(response.json())
        except Exception as e:
            print(f"Error processing audio file: {e}")
        sys.exit(0)

    # Start ngrok tunnel
    public_url = ngrok.connect(config.port, bind_tls=True)
    print(f"ngrok tunnel running at: {public_url}")
    print(f"to join a test room, visit http://localhost:{config.port}/")

    uvicorn.run(
        "server:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
    )