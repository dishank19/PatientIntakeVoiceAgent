#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, PlainTextResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import asyncio
import wave
import shlex

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
from twilio.request_validator import RequestValidator
from loguru import logger
import sys
from src.appointment_store import load_appointments
from datetime import datetime as dt

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

# Pydantic models for the appointments dashboard API (NEWLY ADDED HERE)
class MedicalInfoItem(BaseModel):
    name: Optional[str] = None
    medication: Optional[str] = None
    dosage: Optional[str] = None

class AppointmentDisplayDetail(BaseModel):
    appointment_id: str
    scheduled_datetime_iso: str
    patient_name: Optional[str] = "N/A"
    intent: Optional[str] = "N/A"
    allergies: List[MedicalInfoItem] = []
    prescriptions: List[MedicalInfoItem] = []
    conditions: List[MedicalInfoItem] = []
    twilio_call_sid: Optional[str] = "N/A"

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
        "content": """You are Jessica, the front desk receptionist for a medical clinic called Bay Area Health. Your job is to collect important information from the user before their doctor visit. You should be polite and professional. You're not a medical professional, so you shouldn't provide any advice. Keep your responses short. Your job is to collect information to give to a doctor. Don't make assumptions about what values to plug into functions. Ask for clarification if a user response is ambiguous."""
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

# Initialize Jinja2Templates
# Ensure the "templates" directory is at the same level as server.py or adjust path
# Assuming server.py is at project root, and templates/ is also at project root.
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/")
async def start_agent(request: Request):
    logger.info("GET /: Attempting to start agent with mock SIP call setup")
    try:
        room_url, daily_sip_uri, room_token = await create_daily_room_with_sip()
    except HTTPException as e:
        # HTTPException from create_daily_room_with_sip should be re-raised
        logger.error(f"GET /: HTTPException during Daily SIP room creation: {e.detail}")
        raise e
    except ValueError as ve: # Specific error from create_daily_room_with_sip for config issues
        logger.error(f"GET /: Configuration error during Daily SIP room creation: {ve}")
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        logger.error(f"GET /: Unexpected error during Daily SIP room creation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create SIP-enabled Daily room: {str(e)}"
        )

    logger.info(f"GET /: Daily SIP room created: {room_url}, SIP URI: {daily_sip_uri}")

    if not room_url or not room_token or not daily_sip_uri:
        detail = "Failed to obtain all necessary details (room_url, token, or sip_uri) for SIP-enabled room."
        logger.error(f"GET /: {detail}")
        raise HTTPException(status_code=500, detail=detail)

    # Check if there is already an existing process running in this room
    # Ensure bot_procs access is safe if it can be modified concurrently (not an issue with FastAPI's typical single-process model for dev)
    num_bots_in_room = sum(
        1 for (proc_obj, proc_room_url) in bot_procs.values() if proc_room_url == room_url and proc_obj.poll() is None
    )

    if num_bots_in_room >= MAX_BOTS_PER_ROOM:
        logger.warning(f"GET /: Max bot limit reached for room: {room_url}")
        raise HTTPException(status_code=500, detail=f"Max bot limit reached for room: {room_url}")

    # Generate a mock CallSid for testing via GET request
    mock_call_sid = f"mock-sid-get-{os.urandom(6).hex()}"
    logger.info(f"GET /: Generated mock CallSid: {mock_call_sid}")

    # Spawn a new agent, and join the user session
    try:
        bot_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot.py")
        command_args = [
            sys.executable,  # Use the current Python interpreter
            bot_script_path,
            "-u", room_url,
            "-t", room_token,
            "-cid", mock_call_sid,
            "-s", daily_sip_uri
        ]
        
        logger.info(f"GET /: Launching bot with command: {' '.join(shlex.quote(c) for c in command_args)}")
        
        proc = subprocess.Popen(
            command_args,
            # shell=False by default when args is a list
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            # Consider capturing stdout/stderr for Popen if needed for debugging, similar to create_subprocess_exec
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
        )
        bot_procs[proc.pid] = (proc, room_url)
        logger.info(f"GET /: Bot process started with PID: {proc.pid} for room: {room_url}")

    except Exception as e:
        logger.error(f"GET /: Failed to start bot subprocess: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start bot subprocess: {str(e)}")

    return RedirectResponse(room_url)


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
    form_data = await request.form()
    data = dict(form_data)
    call_sid = data.get('CallSid')
    call_status = data.get('CallStatus')
        
    logger.info(f"Call {call_sid} status update: {call_status}")

    try:
        if call_status == 'completed' or call_status == 'failed' or call_status == 'canceled' or call_status == 'no-answer':
            logger.info(f"Call {call_sid} ended with status: {call_status}. Attempting to find and clean up associated bot process.")
            
            pids_to_remove = []
            # Iterate over a copy of items for safe removal
            for pid, (proc_obj, room_url_assoc) in list(bot_procs.items()):
                # Heuristic: If the bot was launched for this call_sid, we should clean it up.
                # This requires server.py to associate call_sid with the process if it's not directly in bot_procs key.
                # For now, we assume any running bot might be related if not uniquely keyed by call_sid.
                # A more robust solution would be to key bot_procs by call_sid directly if possible.
                
                # Check if the process object is an asyncio.subprocess.Process instance
                if hasattr(proc_obj, 'returncode'): # Characteristic of asyncio.subprocess.Process
                    if proc_obj.returncode is None:  # Process is still running
                        logger.info(f"Bot process PID {pid} (room: {room_url_assoc}) associated with call {call_sid} is still running. Terminating.")
                        try:
                            proc_obj.terminate() # Send SIGTERM
                            # Wait for the process to terminate with a timeout
                            await asyncio.wait_for(proc_obj.wait(), timeout=5.0)
                            logger.info(f"Bot process PID {pid} terminated gracefully.")
                        except asyncio.TimeoutError:
                            logger.warning(f"Timeout waiting for bot process PID {pid} to terminate. Sending SIGKILL.")
                            proc_obj.kill() # Force kill if terminate times out
                            await proc_obj.wait() # Wait for kill to complete
                            logger.info(f"Bot process PID {pid} killed.")
                        except ProcessLookupError:
                            logger.warning(f"Bot process PID {pid} already exited before explicit termination attempt.")
                        except Exception as e_term:
                            logger.error(f"Error during termination of bot process PID {pid}: {e_term}")
                        pids_to_remove.append(pid)
                    else:
                        logger.info(f"Bot process PID {pid} (room: {room_url_assoc}) for call {call_sid} already terminated with code {proc_obj.returncode}. Marking for removal from tracking.")
                        pids_to_remove.append(pid)
                elif hasattr(proc_obj, 'poll'): # Characteristic of subprocess.Popen (older code path)
                    if proc_obj.poll() is None:
                        logger.info(f"Bot process PID {pid} (subprocess.Popen) for call {call_sid} is running. Terminating.")
                        proc_obj.terminate()
                        try:
                            proc_obj.wait(timeout=5.0) # subprocess.Popen.wait
                            logger.info(f"Bot process PID {pid} (subprocess.Popen) terminated.")
                        except subprocess.TimeoutExpired:
                            logger.warning(f"Timeout waiting for bot process PID {pid} (subprocess.Popen) to terminate. Killing.")
                            proc_obj.kill()
                            proc_obj.wait()
                            logger.info(f"Bot process PID {pid} (subprocess.Popen) killed.")
                        except Exception as e_term:
                            logger.error(f"Error terminating bot process PID {pid} (subprocess.Popen): {e_term}")
                        pids_to_remove.append(pid)
                    else:
                        logger.info(f"Bot process PID {pid} (subprocess.Popen) for call {call_sid} already terminated. Marking for removal.")
                        pids_to_remove.append(pid)
                else:
                    logger.warning(f"Bot process PID {pid} in bot_procs is of an unrecognized type. Cannot determine status or terminate.")

            for pid_to_remove in pids_to_remove:
                if pid_to_remove in bot_procs:
                    del bot_procs[pid_to_remove]
                    logger.info(f"Removed bot process PID {pid_to_remove} from tracking.")
            
        return JSONResponse({"status": "success", "message": f"Call status {call_status} for {call_sid} processed."})
        
    except Exception as e:
        logger.error(f"Error handling call status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/twilio/call")
async def handle_twilio_call(request: Request):
    logger.info("Received Twilio call webhook")

    # Validate Twilio request (optional but recommended for production)
    if os.getenv("TWILIO_AUTH_TOKEN"):
        validator = RequestValidator(os.getenv("TWILIO_AUTH_TOKEN"))
        form_ = await request.form()
        url = str(request.url)
        # Twilio sends POST requests with x-www-form-urlencoded content type
        # The signature is in the X-Twilio-Signature header
        signature = request.headers.get("X-Twilio-Signature", "")
        
        # For local testing with ngrok, the URL might be http, but Twilio uses https.
        # Adjust if necessary or ensure ngrok provides https and it matches.
        # If ngrok forwards http to your local https server, this might need care.
        # Assuming request.url is what Twilio used.
        
        # Convert form_ (ImmutableMultiDict) to a dict for validator
        form_params = {key: form_[key] for key in form_}

        if not validator.validate(url, form_params, signature):
            logger.warning("Twilio request validation failed.")
            raise HTTPException(status_code=403, detail="Twilio request validation failed")
        logger.info("Twilio request validated successfully.")
    else:
        logger.warning("TWILIO_AUTH_TOKEN not set, skipping Twilio request validation.")

    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        # caller_number = form_data.get("From")
        # twilio_number = form_data.get("To")
        logger.info(f"Incoming call from Twilio. CallSid: {call_sid}")

        if not call_sid:
            logger.error("CallSid not found in Twilio webhook data.")
            raise HTTPException(status_code=400, detail="CallSid missing from request")

        # 1. Create Daily room with SIP and get token
        try:
            room_url, daily_sip_uri_for_room, room_token = await create_daily_room_with_sip()
            if not daily_sip_uri_for_room:
                logger.error(f"Failed to obtain a valid SIP URI for the Daily room for CallSid {call_sid}. Cannot proceed with bot launch.")
                # Return TwiML indicating an error to Twilio
                response = VoiceResponse()
                response.say("We are currently unable to connect your call. Please try again later.")
                response.hangup()
                return PlainTextResponse(str(response), media_type="application/xml")

        except ValueError as ve: # Raised if DAILY_API_KEY is missing
            logger.error(f"Configuration error: {ve}")
            raise HTTPException(status_code=500, detail=str(ve))
        except HTTPException as http_exc: # Raised by create_daily_room_with_sip on API errors
            logger.error(f"HTTP exception during Daily setup: {http_exc.detail}")
            # Return TwiML indicating an error to Twilio
            response = VoiceResponse()
            response.say("There was an issue setting up the call environment. Please try again later.")
            response.hangup()
            return PlainTextResponse(str(response), media_type="application/xml")


        # 2. Start bot.py as a subprocess
        # Ensure bot.py is executable and path is correct
        bot_script_path = os.path.join(os.path.dirname(__file__), "bot.py")
        
        # Pass necessary details to bot.py
        # The -s argument to bot.py is the SIP URI it should instruct Twilio to dial.
        # This SIP URI is for the Daily room we just created.
        command = [
            sys.executable,  # Path to python interpreter
            bot_script_path,
            "-u", room_url,
            "-t", room_token,
            "-cid", call_sid, # Twilio CallSid
            "-s", daily_sip_uri_for_room # The SIP URI of the Daily room for Twilio to dial
        ]
        
        logger.info(f"Launching bot: {' '.join(shlex.quote(c) for c in command)}")
        
        # Start the bot process
        # Store the process to manage its lifecycle if needed (e.g., on server shutdown)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=subprocess.PIPE, # Or subprocess.DEVNULL if you don't need to capture
            stderr=subprocess.PIPE  # Or subprocess.DEVNULL
        )
        bot_procs[process.pid] = (process, room_url)
        logger.info(f"Bot process started for CallSid: {call_sid} with PID: {process.pid}")
        
        # Non-blocking monitoring of the bot's stdout/stderr
        async def log_stream(stream, stream_name):
            while True:
                line = await stream.readline()
                if not line:
                    break
                logger.info(f"Bot ({stream_name}) CallSid {call_sid}: {line.decode().strip()}")

        asyncio.create_task(log_stream(process.stdout, "stdout"))
        asyncio.create_task(log_stream(process.stderr, "stderr"))


        # 3. Respond to Twilio to put the caller on hold (or play a message)
        # The actual call forwarding to SIP will be done by bot.py via Twilio API update
        response = VoiceResponse()
        # Example: Play hold music or a waiting message
        response.say("Thank you for calling Tri-County Health Services. Please wait while we connect you to an agent.")
        # response.play(url="YOUR_HOLD_MUSIC_URL.mp3") # If you have hold music
        # Keep the call alive; bot.py will update it.
        # A long pause can keep the line open until the bot updates the call.
        # Twilio might timeout if it doesn't receive further TwiML or actions.
        # A common pattern is to <Enqueue> into a queue, and the bot (or another process)
        # then dequeues and bridges. But for direct update, a long <Pause> works.
        # If the bot is quick, this might be fine. Otherwise, <Enqueue> is more robust.
        # For simplicity now, let's use a long Pause. The bot should update well before this.
        response.pause(length=60) # Pause for 60 seconds, bot should update call before this.
        # If the bot fails to update, the call will hang up after this pause.

        logger.info(f"Responding to Twilio for CallSid {call_sid} with TwiML: {str(response)}")
        return PlainTextResponse(str(response), media_type="application/xml")

    except Exception as e:
        logger.exception(f"Unhandled error in /webhook/twilio/call for CallSid {form_data.get('CallSid', 'UNKNOWN')}")
        # Generic error response to Twilio
        response = VoiceResponse()
        response.say("An unexpected error occurred. Please try your call again later.")
        response.hangup()
        return PlainTextResponse(str(response), media_type="application/xml")

async def create_daily_room_with_sip() -> tuple[str, str, str]:
    """
    Creates a Daily room with SIP enabled and returns (room_url, room_sip_uri, room_token).
    The room_token is for the bot to join with owner privileges.
    """
    if not os.getenv("DAILY_API_KEY"):
        raise ValueError("DAILY_API_KEY not set in environment variables.")
    headers = {"Authorization": f"Bearer {os.getenv('DAILY_API_KEY')}"}
    
    async with aiohttp.ClientSession(headers=headers) as session:
        now = datetime.utcnow()
        exp = int((now + timedelta(hours=1)).timestamp())
        
        room_properties = {
            "exp": exp,
            "eject_at_room_exp": True,
            "sip": {
                "sip_mode": "dial-in",
                "display_name": "HealthIntakeBot"
            }
        }
        
        logger.info(f"[DEBUG] Sending room_properties to Daily API: {json.dumps(room_properties)}") # Log the properties

        async with session.post(f"{os.getenv('DAILY_API_URL', 'https://api.daily.co/v1')}/rooms", 
                                json={"properties": room_properties}) as resp_room:
            if resp_room.status != 200:
                error_text = await resp_room.text()
                # Log the detailed error text from Daily before raising HTTPException
                logger.error(f"Error creating Daily room. Status: {resp_room.status}, Response: {error_text}, Sent properties: {json.dumps(room_properties)}")
                raise HTTPException(status_code=500, detail=f"Error creating Daily room: {error_text}")
            
            room_data = await resp_room.json()
            room_url = room_data.get("url")
            room_name = room_data.get("name")
            logger.info(f"Full Daily room creation response: {json.dumps(room_data)}") # Log the full response
            logger.info(f"Daily room created: {room_url} (Name: {room_name})")

            if not room_url or not room_name:
                logger.error(f"Daily room creation response missing URL or name: {room_data}")
                raise HTTPException(status_code=500, detail="Failed to get valid room URL/name from Daily.")

            daily_sip_uri_for_room = None
            
            # Primary Attempt: Extract from config.sip_uri.endpoint based on observed response
            config_obj = room_data.get("config", {})
            if isinstance(config_obj, dict):
                sip_uri_obj = config_obj.get("sip_uri", {})
                if isinstance(sip_uri_obj, dict):
                    daily_sip_uri_for_room = sip_uri_obj.get("endpoint")

            if daily_sip_uri_for_room:
                logger.info(f"Extracted Daily SIP URI (from config.sip_uri.endpoint) for room {room_name}: {daily_sip_uri_for_room}")
            else:
                logger.info(f"SIP URI not found in config.sip_uri.endpoint for room {room_name}. Trying fallback locations.")
                
                # Fallback 1: Check config.sip_endpoint (less likely, but keeping for safety for a moment)
                if isinstance(config_obj, dict): # Ensure config_obj is a dictionary (already checked but good for clarity)
                    fallback_sip_endpoint = config_obj.get("sip_endpoint")
                    if fallback_sip_endpoint:
                        daily_sip_uri_for_room = fallback_sip_endpoint
                        logger.info(f"Extracted Daily SIP URI (from FALLBACK config.sip_endpoint) for room {room_name}: {daily_sip_uri_for_room}")
                    else:
                        logger.info(f"SIP URI also not found in FALLBACK config.sip_endpoint for room {room_name}.")

                # Fallback 2: Check top-level 'sip' object (even less likely given current response structure)
                if not daily_sip_uri_for_room: # Only proceed if still not found
                    logger.info(f"Attempting FALLBACK to top-level 'sip' object for room {room_name}. This is less likely to succeed based on current logs.")
                    sip_info = room_data.get("sip", {}) # Use .get with default {} for safety
                    if isinstance(sip_info, dict):
                        if isinstance(sip_info.get("endpoints"), list) and sip_info["endpoints"]:
                            try:
                                for endpoint in sip_info["endpoints"]:
                                    if endpoint.get("url"):
                                        daily_sip_uri_for_room = endpoint["url"]
                                        break
                                if daily_sip_uri_for_room:
                                    logger.info(f"Extracted Daily SIP URI (from FALLBACK sip.endpoints) for room {room_name}: {daily_sip_uri_for_room}")
                                else:
                                    logger.warning(f"SIP URI string not found in FALLBACK sip.endpoints for {room_name}. Data: {sip_info}")
                            except Exception as e:
                                logger.warning(f"Could not extract SIP URI from FALLBACK sip.endpoints for {room_name}: {e}. Data: {sip_info}")
                        elif isinstance(sip_info.get("uris"), list) and sip_info["uris"]:
                            try:
                                daily_sip_uri_for_room = sip_info["uris"][0].get("uri")
                                if daily_sip_uri_for_room:
                                    logger.info(f"Extracted Daily SIP URI (from FALLBACK sip.uris) for room {room_name}: {daily_sip_uri_for_room}")
                                else:
                                    logger.warning(f"SIP URI string is empty in FALLBACK sip.uris for {room_name}. Data: {sip_info}")
                            except (IndexError, KeyError, AttributeError) as e:
                                logger.warning(f"Could not extract SIP URI from FALLBACK sip.uris for {room_name}: {e}. Data: {sip_info}")
                        else:
                            logger.warning(f"No SIP URIs or endpoints found in FALLBACK 'sip' object for {room_name}. SIP Info: {sip_info}")
                    else:
                        logger.warning(f"FALLBACK 'sip' key not found in room_data or is not a dictionary for {room_name}. Room Data Keys: {list(room_data.keys())}")

            if not daily_sip_uri_for_room:
                logger.error(f"CRITICAL: Could not determine a specific SIP URI for room {room_name}. Twilio forwarding by bot will fail.")
                raise HTTPException(status_code=500, detail=f"Could not determine SIP URI for Daily room {room_name}. Bot cannot operate.")

        token_payload = {
            "properties": {
                "room_name": room_name, 
                "is_owner": True,
                "user_name": "HealthIntakeBotInstance",
                "exp": exp 
            }
        }
        async with session.post(f"{os.getenv('DAILY_API_URL', 'https://api.daily.co/v1')}/meeting-tokens", json=token_payload) as resp_token:
            if resp_token.status != 200:
                error_text = await resp_token.text()
                logger.error(f"Error creating Daily meeting token: {resp_token.status} {error_text}")
                raise HTTPException(status_code=500, detail=f"Error creating Daily meeting token: {error_text}")
            token_data = await resp_token.json()
            room_token = token_data.get("token")
            if not room_token:
                logger.error(f"Daily meeting token creation response missing token: {token_data}")
                raise HTTPException(status_code=500, detail="Failed to get valid room token from Daily.")
            logger.info(f"Daily meeting token generated for room: {room_name}")

    return room_url, daily_sip_uri_for_room, room_token

@app.get("/dashboard/appointments", response_class=HTMLResponse)
async def get_appointments_dashboard(request: Request):
    logger.info("GET /dashboard/appointments: Serving appointments dashboard HTML.")
    return templates.TemplateResponse("appointments.html", {"request": request})

@app.get("/api/appointments/{date_str}", response_model=List[AppointmentDisplayDetail])
async def get_appointments_for_date(date_str: str):
    logger.info(f"GET /api/appointments/{date_str}: Fetching appointments.")
    try:
        target_date = dt.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        logger.error(f"Invalid date format for /api/appointments: {date_str}")
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    all_appointments = load_appointments() # From src.appointment_store
    appointments_for_date: List[AppointmentDisplayDetail] = []

    for appt in all_appointments:
        try:
            appt_datetime_iso = appt.get("scheduled_datetime_iso")
            if not appt_datetime_iso:
                logger.warning(f"Appointment {appt.get('appointment_id')} missing scheduled_datetime_iso, skipping.")
                continue
            
            appt_dt = dt.fromisoformat(appt_datetime_iso)
            if appt_dt.date() == target_date:
                # Convert allergy, prescription, condition data to MedicalInfoItem model
                # The data in appointments.json might be a list of strings or simple objects
                # The HTML expects objects like {'name': 'xyz'} or {'medication': 'abc', 'dosage': '123'}
                
                def _map_medical_list(data_list, type_key_primary, type_key_secondary=None):
                    mapped_list = []
                    if isinstance(data_list, list):
                        for item in data_list:
                            if isinstance(item, dict):
                                if type_key_secondary:
                                     mapped_list.append(MedicalInfoItem(**{type_key_primary: item.get(type_key_primary), type_key_secondary: item.get(type_key_secondary)}))
                                else:
                                    mapped_list.append(MedicalInfoItem(**{type_key_primary: item.get(type_key_primary)}))
                            elif isinstance(item, str):
                                mapped_list.append(MedicalInfoItem(**{type_key_primary: item})) # Store string directly under primary key
                    return mapped_list

                appointments_for_date.append(
                    AppointmentDisplayDetail(
                        appointment_id=appt.get("appointment_id", "N/A"),
                        scheduled_datetime_iso=appt_datetime_iso,
                        patient_name=appt.get("patient_name"),
                        intent=appt.get("intent"),
                        allergies=_map_medical_list(appt.get("allergies", []), "name"),
                        prescriptions=_map_medical_list(appt.get("prescriptions", []), "medication", "dosage"),
                        conditions=_map_medical_list(appt.get("conditions", []), "name"),
                        twilio_call_sid=appt.get("twilio_call_sid")
                    )
                )
        except Exception as e:
            logger.error(f"Error processing appointment {appt.get('appointment_id', 'UNKNOWN')} for API: {e}")
            # Optionally skip this appointment or handle error differently
            continue
    
    logger.info(f"Returning {len(appointments_for_date)} appointments for date {date_str}.")
    return appointments_for_date

if __name__ == "__main__":
    if not os.getenv("DAILY_API_KEY"):
        logger.critical("CRITICAL: DAILY_API_KEY is not set in environment variables. The server will not be able to create Daily rooms.")
        # sys.exit(1) # Exit if critical env var is missing
    
    # Get host and port from environment or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    
    logger.info(f"Attempting to start Uvicorn server on host {host} and port {port}")
    logger.info("Uvicorn will provide the final accessible URL(s) upon successful startup (typically http://<host>:<port>).")
    
    # For production, consider using a more robust ASGI server like Gunicorn with Uvicorn workers
    # uvicorn.run(app, host=host, port=port)
    # Using app="server:app" to allow uvicorn to find the app instance when run directly
    uvicorn.run("server:app", host=host, port=port, reload=True) # reload=True for dev