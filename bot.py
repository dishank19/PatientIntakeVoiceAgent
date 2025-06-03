#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
import wave
import json
from datetime import datetime
import argparse

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.logger import FrameLogger
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMContext, OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

sounds = {}
sound_files = [
    "clack-short.wav",
    "clack.wav",
    "clack-short-quiet.wav",
    "ding.wav",
    "ding2.wav",
]

script_dir = os.path.dirname(__file__)

for file in sound_files:
    # Build the full path to the sound file
    full_path = os.path.join(script_dir, "assets", file)
    # Get the filename without the extension to use as the dictionary key
    filename = os.path.splitext(os.path.basename(full_path))[0]
    # Open the sound and convert it to bytes
    with wave.open(full_path) as audio_file:
        sounds[file] = OutputAudioRawFrame(
            audio_file.readframes(-1), audio_file.getframerate(), audio_file.getnchannels()
        )


class IntakeProcessor:
    def __init__(self, context: OpenAILLMContext):
        print(f"Initializing context from IntakeProcessor")
        # Initialize call data dictionary
        self.call_data = {}
        self.is_spanish = False
        self.context = context  # Store context for access in handlers
        self.pipeline = None  # Will be set when pipeline is created
        context.add_message(
            {
                "role": "system",
                "content": """You are Jessica, an agent for a company called Tri-County Health Services. Your job is to collect important information from the user before their doctor visit. You should be polite and professional. You're not a medical professional, so you shouldn't provide any advice. Keep your responses short. Your job is to collect information to give to a doctor. Don't make assumptions about what values to plug into functions. Ask for clarification if a user response is ambiguous.

Important conversation guidelines:
1. After each question, wait for the user to finish speaking completely before responding
2. If the user indicates they have nothing else to add (e.g., "that's all", "no thank you", "that's everything"), end the call politely
3. If ending the call, mention looking forward to their visit if they have an appointment scheduled
4. Use natural pauses in your speech to make the conversation feel more natural
5. If you detect the user speaking Spanish, switch to Spanish and continue the conversation in Spanish while maintaining the same information collection flow

Start by introducing yourself and asking for the patient's name. Then, ask what brings them in today. Based on their response, classify their intent and call the classify_intent function.""",
            }
        )
        context.set_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "classify_intent",
                        "description": "Classify the intent of the patient's visit based on their response.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "patient_name": {
                                    "type": "string",
                                    "description": "The patient's full name",
                                },
                                "intent": {
                                    "type": "string",
                                    "description": "The classified intent of the visit (e.g., appointment_scheduling, billing_inquiry, prescription_refill, general_inquiry, emergency, follow_up)",
                                },
                                "details": {
                                    "type": "string",
                                    "description": "Additional details about the intent",
                                },
                                "is_spanish": {
                                    "type": "boolean",
                                    "description": "Whether the user is speaking Spanish",
                                }
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "end_call",
                        "description": "End the call when the user indicates they have nothing else to add.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "has_appointment": {
                                    "type": "boolean",
                                    "description": "Whether the patient has an appointment scheduled",
                                },
                                "appointment_date": {
                                    "type": "string",
                                    "description": "The date of the appointment if scheduled",
                                }
                            },
                        },
                    },
                }
            ]
        )

    async def classify_intent(self, params: FunctionCallParams):
        # Store initial data
        self.call_data.update(params.arguments)
        self.is_spanish = params.arguments.get("is_spanish", False)
        
        # Move on to prescriptions
            params.context.set_tools(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_prescriptions",
                            "description": "Once the user has provided a list of their prescription medications, call this function.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "prescriptions": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "medication": {
                                                    "type": "string",
                                                    "description": "The medication's name",
                                                },
                                                "dosage": {
                                                    "type": "string",
                                                    "description": "The prescription's dosage",
                                                },
                                            },
                                        },
                                    }
                                },
                            },
                        },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "end_call",
                        "description": "End the call when the user indicates they have nothing else to add.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "has_appointment": {
                                    "type": "boolean",
                                    "description": "Whether the patient has an appointment scheduled",
                                },
                                "appointment_date": {
                                    "type": "string",
                                    "description": "The date of the appointment if scheduled",
                                }
                            },
                        },
                    },
                }
            ]
        )
        
        if self.is_spanish:
            await params.result_callback(
                [
                    {
                        "role": "system",
                        "content": "Gracias por esa información. <break time='1s'/> ¿Podría decirme qué medicamentos está tomando actualmente?",
                    }
                ]
            )
        else:
            await params.result_callback(
                [
                    {
                        "role": "system",
                        "content": "Thanks for that information. <break time='1s'/> Could you tell me what medications you're currently taking?",
                    }
                ]
            )

    async def list_prescriptions(self, params: FunctionCallParams):
        # Store prescriptions data
        self.call_data.update(params.arguments)
        
        # Move on to allergies
        params.context.set_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "list_allergies",
                        "description": "Once the user has provided a list of their allergies, call this function.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "allergies": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {
                                                "type": "string",
                                                "description": "What the user is allergic to",
                                            }
                                        },
                                    },
                                }
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "end_call",
                        "description": "End the call when the user indicates they have nothing else to add.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "has_appointment": {
                                    "type": "boolean",
                                    "description": "Whether the patient has an appointment scheduled",
                                },
                                "appointment_date": {
                                    "type": "string",
                                    "description": "The date of the appointment if scheduled",
                                }
                            },
                        },
                    },
                }
            ]
        )
        
        if self.is_spanish:
            params.context.add_message(
                {
                    "role": "system",
                    "content": "¿Y hay algo a lo que sea alérgico?",
                }
            )
        else:
        params.context.add_message(
            {
                "role": "system",
                    "content": "And is there anything you're allergic to?",
            }
        )
        await params.llm.queue_frame(
            OpenAILLMContextFrame(params.context), FrameDirection.DOWNSTREAM
        )

    async def list_allergies(self, params: FunctionCallParams):
        # Store allergies data
        self.call_data.update(params.arguments)
        
        # Move on to conditions
        params.context.set_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "list_conditions",
                        "description": "Once the user has provided a list of their medical conditions, call this function.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "conditions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {
                                                "type": "string",
                                                "description": "The user's medical condition",
                                            }
                                        },
                                    },
                                }
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "end_call",
                        "description": "End the call when the user indicates they have nothing else to add.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "has_appointment": {
                                    "type": "boolean",
                                    "description": "Whether the patient has an appointment scheduled",
                                },
                                "appointment_date": {
                                    "type": "string",
                                    "description": "The date of the appointment if scheduled",
                                }
                            },
                        },
                    },
                }
            ]
        )
        
        if self.is_spanish:
            params.context.add_message(
                {
                    "role": "system",
                    "content": "¿Hay alguna condición médica que el doctor deba conocer?",
                }
            )
        else:
        params.context.add_message(
            {
                "role": "system",
                    "content": "Is there any medical condition the doctor should know about?",
            }
        )
        await params.llm.queue_frame(
            OpenAILLMContextFrame(params.context), FrameDirection.DOWNSTREAM
        )

    async def list_conditions(self, params: FunctionCallParams):
        # Store conditions data
        self.call_data.update(params.arguments)
        
        # Move on to visit reasons
        params.context.set_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "list_visit_reasons",
                        "description": "Once the user has provided a list of the reasons they are visiting a doctor today, call this function.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "visit_reasons": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {
                                                "type": "string",
                                                "description": "The user's reason for visiting the doctor",
                                            }
                                        },
                                    },
                                }
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "end_call",
                        "description": "End the call when the user indicates they have nothing else to add.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "has_appointment": {
                                    "type": "boolean",
                                    "description": "Whether the patient has an appointment scheduled",
                                },
                                "appointment_date": {
                                    "type": "string",
                                    "description": "The date of the appointment if scheduled",
                                }
                            },
                        },
                    },
                }
            ]
        )
        
        if self.is_spanish:
            params.context.add_message(
                {
                    "role": "system",
                    "content": "¿Y qué le trae hoy a la consulta?",
                }
            )
        else:
        params.context.add_message(
            {
                "role": "system",
                    "content": "And what brings you in today?",
            }
        )
        await params.llm.queue_frame(
            OpenAILLMContextFrame(params.context), FrameDirection.DOWNSTREAM
        )

    async def list_visit_reasons(self, params: FunctionCallParams):
        # Store visit reasons data
        self.call_data.update(params.arguments)
        
        # Save the complete call data
        await self.save_data(self.call_data, params.result_callback)
        
        if self.is_spanish:
            await params.result_callback(
                [
                    {
                        "role": "system",
                        "content": "Gracias por toda esta información. <break time='1s'/> El doctor la revisará antes de su visita. ¿Hay algo más que quiera mencionar?",
                    }
                ]
            )
        else:
            await params.result_callback(
                [
                    {
                        "role": "system",
                        "content": "Thanks for all this information. <break time='1s'/> The doctor will review it before your visit. Is there anything else you'd like to mention?",
                    }
                ]
            )

    async def end_call(self, params: FunctionCallParams):
        # Store appointment information if available
        self.call_data.update(params.arguments)
        
        # Save the complete call data
        await self.save_data(self.call_data, params.result_callback)
        
        # Generate appropriate closing message
        if self.is_spanish:
            if params.arguments.get("has_appointment", False):
                appointment_date = params.arguments.get("appointment_date", "")
                closing_message = f"Gracias por su tiempo. <break time='1s'/> Esperamos verle el {appointment_date}. ¡Que tenga un excelente día!"
            else:
                closing_message = "Gracias por su tiempo. <break time='1s'/> ¡Que tenga un excelente día!"
        else:
            if params.arguments.get("has_appointment", False):
                appointment_date = params.arguments.get("appointment_date", "")
                closing_message = f"Thank you for your time. <break time='1s'/> We look forward to seeing you on {appointment_date}. Have a wonderful day!"
            else:
                closing_message = "Thank you for your time. <break time='1s'/> Have a wonderful day!"
        
        await params.result_callback(
            [
                {
                    "role": "system",
                    "content": closing_message,
                }
            ]
        )
        
        # Exit the room after a short delay to allow the closing message to be delivered
        await asyncio.sleep(2)
        if self.pipeline and self.pipeline.transport:
            await self.pipeline.transport.leave()

    async def save_data(self, args, result_callback):
        try:
            # Get the current date and time
            current_time = datetime.now()
            date_str = current_time.strftime("%Y%m%d_%H%M%S")
            
            # Get patient name and intent from the arguments if available
            patient_name = args.get("patient_name", "unknown_patient")
            intent = args.get("intent", "general_inquiry")
            
            # Add call status to the data
            args["call_status"] = "completed" if "call_status" not in args else args["call_status"]
            
            # Create filename
            filename = f"{date_str}_{patient_name}_{intent}.json"
            
            # Create data directory if it doesn't exist
            data_dir = os.path.join(script_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            # Save the data
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "w") as f:
                json.dump(args, f, indent=2)
                
            print(f"Saved complete call data to {filepath}")
        except Exception as e:
            print(f"Error saving data: {e}")


async def main(room_url: str, token: str, call_id: str, sip_uri: str):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Health Intake Bot",
            DailyParams(
                api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
                api_key=os.getenv("DAILY_API_KEY", ""),
                audio_in_enabled=True,
                audio_out_enabled=True,
                camera_out_enabled=False,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                transcription_enabled=True,
            )
        )

        # Initialize TTS service based on detected language
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="846d6cb0-2301-48b6-9683-48f5618ea2f6",  # Spanish-speaking Lady
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        messages = []
        context = OpenAILLMContext(messages=messages)
        context_aggregator = llm.create_context_aggregator(context)

        intake = IntakeProcessor(context)
        llm.register_function("classify_intent", intake.classify_intent)
        llm.register_function("list_prescriptions", intake.list_prescriptions)
        llm.register_function("list_allergies", intake.list_allergies)
        llm.register_function("list_conditions", intake.list_conditions)
        llm.register_function("list_visit_reasons", intake.list_visit_reasons)
        llm.register_function("end_call", intake.end_call)

        fl = FrameLogger("LLM Output")

        pipeline = Pipeline(
            [
                transport.input(),  # Transport input
                context_aggregator.user(),  # User responses
                llm,  # LLM
                fl,  # Frame logger
                tts,  # TTS
                transport.output(),  # Transport output
                context_aggregator.assistant(),  # Assistant responses
            ]
        )

        # Store pipeline reference in intake processor
        intake.pipeline = pipeline
        
        # Create pipeline task
        task = PipelineTask(pipeline, params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ))
        intake.pipeline.task = task

        @transport.event_handler("on_dialin_ready")
        async def on_dialin_ready(transport, cdata):
            try:
                # Update Twilio call to connect to Daily SIP endpoint
                call = twilio_client.calls(call_id).update(
                    twiml=f'<Response><Dial><Sip>{sip_uri}</Sip></Dial></Response>'
                )
                logger.info(f"Call {call_id} forwarded to Daily SIP endpoint")
            except Exception as e:
                logger.error(f"Failed to forward call: {str(e)}")
                raise

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([OpenAILLMContextFrame(context)])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, *args):
            # Save data when participant leaves
            await intake.save_data({}, None)

        @transport.event_handler("on_interruption")
        async def on_interruption(transport, participant, *args):
            # Handle interruptions gracefully
            logger.info(f"Interruption detected from participant {participant['id']}")

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Daily Example")
    parser.add_argument("-u", type=str, help="Room URL")
    parser.add_argument("-t", type=str, help="Token")
    parser.add_argument("-i", type=str, help="Call ID")
    parser.add_argument("-s", type=str, help="SIP URI")
    config = parser.parse_args()

    asyncio.run(main(config.u, config.t, config.i, config.s))