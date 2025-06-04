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
from datetime import datetime, timedelta, time as dt_time
import argparse
import random # Added for suggesting random minutes
import uuid # Added uuid
from typing import Optional, List, Dict, Any # Added Optional, List, Dict, Any

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure
from twilio.rest import Client # Added Twilio client

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import OutputAudioRawFrame, EndTaskFrame # Added EndTaskFrame
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
from src.appointment_store import load_appointments, save_appointments # New import

# Attempt to import dateutil.parser, with a fallback or warning if not available.
# This should be in requirements.txt
try:
    from dateutil import parser as dateutil_parser
    from dateutil.relativedelta import relativedelta
except ImportError:
    logger.error("python-dateutil is not installed. Please install it: pip install python-dateutil")
    # Fallback to a dummy parser or raise an error if dateutil is critical
    dateutil_parser = None 
    relativedelta = None

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

# Define business hours and slot duration (can be moved to config or class attributes)
BUSINESS_START_HOUR = 9  # 9 AM
BUSINESS_END_HOUR = 17   # 5 PM (last slot starts before 5 PM)
APPOINTMENT_DURATION_MINUTES = 30

class IntakeProcessor:
    def __init__(self, context: OpenAILLMContext, twilio_call_sid: str):
        print(f"Initializing context from IntakeProcessor for CallSid: {twilio_call_sid}")
        self.twilio_call_sid = twilio_call_sid  # Store the CallSid
        self.call_data = {}
        self.is_spanish = False
        self.context = context
        self.pipeline = None
        self.call_data["patient_name"] = None # Initialize to ensure it exists
        self.call_data["intent"] = None       # Initialize to ensure it exists
        context.add_message(
            {
                "role": "system",
                "content": """You are Jessica, an agent for Bay Area Health. Your job is to collect important information and assist with appointment scheduling. Be polite, professional, and keep responses natural. You are not a medical professional.

Important conversation guidelines:
1. Start by introducing yourself and asking for the patient's name.
2. Wait for the user to finish speaking before responding. Use <break time='1s'/> for noticeable pauses.
3. Let the patient guide the conversation.
4. If the patient mentions medications, allergies, or conditions, use 'collect_medical_info'.
5. If they don't mention these, and the main interaction (like scheduling) is concluding, ask once if there's anything else medical the doctor should know.
6. For appointment requests:
    - Use the 'end_call' function with action 'request_schedule'. Provide 'requested_date_time' (e.g., "tomorrow 3 PM") AND 'requested_hour_24_format' (e.g., 15 for 3 PM).
    - If a proposed time is confirmed by the patient, call 'end_call' with action 'confirm_schedule' and 'is_appointment_confirmed_by_patient': true. Include the confirmed 'requested_date_time' and 'requested_hour_24_format'.
    - If a proposed time is declined, or the patient wants a different time, call 'end_call' with action 'request_schedule' and the new 'requested_date_time' and 'requested_hour_24_format'.
    - If the patient wants to stop scheduling after a decline, call 'end_call' with action 'terminate_interaction'.
7. When the conversation is truly over, or after a successful booking and no further questions, call 'end_call' with action 'terminate_interaction'. If an appointment was booked, provide 'final_appointment_details_for_goodbye'.
8. If Spanish is detected, switch to Spanish for all interactions.
""",
            }
        )
        context.set_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "classify_intent",
                        "description": "Classify the initial intent of the patient's visit after they provide their name and reason for calling.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "patient_name": {"type": "string", "description": "The patient's full name"},
                                "intent": {"type": "string", "description": "Classified intent (e.g., appointment_scheduling, billing_inquiry, general_inquiry)"},
                                "details": {"type": "string", "description": "Additional details about the intent"},
                                "is_spanish": {"type": "boolean", "description": "Is the user speaking Spanish?"}
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "collect_medical_info",
                        "description": "Collect medical information (prescriptions, allergies, conditions) if mentioned by the patient or if appropriate to ask.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "prescriptions": {"type": "array", "items": {"type": "object", "properties": {"medication": {"type": "string"}, "dosage": {"type": "string"}}}},
                                "allergies": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}}}},
                                "conditions": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}}}}
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "end_call",
                        "description": "Handles appointment scheduling logic (requests, availability checks, confirmations) or concludes the call.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["request_schedule", "confirm_schedule", "terminate_interaction"],
                                    "description": "The specific action: 'request_schedule' to check/propose a time, 'confirm_schedule' to finalize or decline a proposed time, 'terminate_interaction' to end the call."
                                },
                                "requested_date_time": {
                                    "type": "string",
                                    "description": "The full date and time string for an appointment request (e.g., 'tomorrow 3 PM', 'next Tuesday at 10 AM'). Used with 'request_schedule'."
                                },
                                "requested_hour_24_format": {
                                    "type": "integer",
                                    "description": "The hour (0-23) of the 'requested_date_time'. Essential for availability check. Used with 'request_schedule'."
                                },
                                "is_appointment_confirmed_by_patient": {
                                    "type": "boolean",
                                    "description": "Set to true if the patient confirms a proposed/checked appointment slot. Set to false if they decline it. Used with 'confirm_schedule'."
                                },
                                "final_appointment_details_for_goodbye": {
                                    "type": "string",
                                    "description": "If an appointment was successfully booked, this is the confirmed time string (e.g. 'tomorrow 3 PM') to be mentioned in the closing statement. Only use with 'terminate_interaction' after a successful booking."
                                }
                            },
                            "required": ["action"]
                        }
                    }
                }
            ]
        )

    async def classify_intent(self, params: FunctionCallParams):
        self.call_data.update(params.arguments)
        self.is_spanish = params.arguments.get("is_spanish", False)
        patient_name = params.arguments.get("patient_name", "there")

        intent_provided = params.arguments.get("intent")
        
        response_message = ""

        if not intent_provided:
            if self.is_spanish:
                response_message = f"Gracias, {patient_name}. <break time='1s'/> ¿Y cuál es el motivo de su llamada hoy?"
            else:
                response_message = f"Thanks, {patient_name}. <break time='1s'/> And what is the reason for your call today?"
            # Tools remain as initially set (classify_intent, collect_medical_info, end_call)
        else:
            self.call_data["intent"] = intent_provided
            if params.arguments.get("details"):
                self.call_data["details"] = params.arguments.get("details")

            # Narrow tools if intent is now clear and not scheduling. 
            # If intent IS scheduling, LLM will use end_call with action "request_schedule".
            current_tools_config = []
            # Keep classify_intent available if LLM needs to re-classify after more info.
            if "classify_intent" in self.context._tools_by_name:
                 current_tools_config.append(self.context._tools_by_name["classify_intent"].config)
            if "collect_medical_info" in self.context._tools_by_name:
                current_tools_config.append(self.context._tools_by_name["collect_medical_info"].config)
            if "end_call" in self.context._tools_by_name:
                current_tools_config.append(self.context._tools_by_name["end_call"].config)
            
            if current_tools_config:
                params.context.set_tools(current_tools_config)
        
        if self.is_spanish:
                response_message = f"Entendido. Usted se contactó por {self.call_data.get('intent', 'su consulta')}. <break time='1s'/> ¿Hay algo más en lo que pueda ayudarle o alguna otra información que desee compartir?"
        else:
                response_message = f"Okay, I understand you're calling about {self.call_data.get('intent', 'your inquiry')}. <break time='1s'/> Is there anything else I can help you with, or any other information you'd like to share?"
        
        await params.result_callback([{"role": "system", "content": response_message}])

    def _list_to_natural_language(self, items: list[str], lang_map: dict, conjunction: str) -> str:
        if not items:
            return ""
        
        mapped_items = [lang_map.get(item, item) for item in items]

        if len(mapped_items) == 1:
            return mapped_items[0]
        
        if len(mapped_items) == 2:
            return f"{mapped_items[0]} {conjunction} {mapped_items[1]}"
        
        # For 3 or more items, use a comma-separated list with the conjunction before the last item
        return f"{', '.join(mapped_items[:-1])} {conjunction} {mapped_items[-1]}"

    async def collect_medical_info(self, params: FunctionCallParams):
        self.call_data.update(params.arguments)

        # Initialize tracking in self.call_data if not present
        if "medical_topics_to_cover" not in self.call_data:
            self.call_data["medical_topics_to_cover"] = ["allergies", "conditions", "prescriptions"]
        if "medical_info_initial_ask_done" not in self.call_data:
            self.call_data["medical_info_initial_ask_done"] = False
        if "medical_info_follow_up_done" not in self.call_data:
            self.call_data["medical_info_follow_up_done"] = False

        # Identify what information was provided in this specific turn by the LLM
        newly_provided_topics_this_turn = []
        topics_to_check = ["allergies", "conditions", "prescriptions"]
        for topic in topics_to_check:
            # Check if the LLM provided a non-empty list/array for the topic
            if params.arguments.get(topic) and isinstance(params.arguments.get(topic), list) and len(params.arguments.get(topic)) > 0:
                newly_provided_topics_this_turn.append(topic)
                if topic in self.call_data["medical_topics_to_cover"]:
                    self.call_data["medical_topics_to_cover"].remove(topic)
        
        response_message = ""

        # Language strings
        topic_map_es = {"allergies": "alergias", "conditions": "condiciones médicas", "prescriptions": "prescripciones"}
        topic_map_en = {"allergies": "allergies", "conditions": "medical conditions", "prescriptions": "prescriptions"}
        
        lang_and_es = "y"
        lang_and_en = "and"
        lang_or_es = "o" # Used for questions about remaining items
        lang_or_en = "or"

        current_topic_map = topic_map_es if self.is_spanish else topic_map_en
        current_and_conjunction = lang_and_es if self.is_spanish else lang_and_en
        current_or_conjunction = lang_or_es if self.is_spanish else lang_or_en

        lang_initial_medical_q_es = "¿Para asegurarnos de tener todos los detalles, podría informarme sobre cualquier prescripción actual, alergia o condición médica preexistente que el médico deba conocer?"
        lang_initial_medical_q_en = "To ensure we have all details, could you tell me about any current prescriptions, allergies, or existing medical conditions the doctor should be aware of?"
        
        lang_thank_you_for_info_es = "Gracias por la información sobre sus"
        lang_thank_you_for_info_en = "Thanks for the information about your"
        lang_thank_you_general_es = "Gracias por compartir eso."
        lang_thank_you_general_en = "Thanks for sharing that."

        lang_follow_up_intro_es = "Solo para estar seguros,"
        lang_follow_up_intro_en = "Just to be sure,"
        
        lang_follow_up_query_es = "¿podría también informarme sobre"
        lang_follow_up_query_en = "could you also tell me about any"

        lang_anything_else_medical_es = "¿Hay algo más que desee agregar sobre estos temas médicos?"
        lang_anything_else_medical_en = "Is there anything else you'd like to add on these medical topics?"
        
        lang_generic_thanks_and_continue_es = "Entendido, gracias. ¿Hay algo más en lo que pueda ayudarle?"
        lang_generic_thanks_and_continue_en = "Okay, thank you. Is there anything else I can help with?"

        if not self.call_data["medical_info_initial_ask_done"]:
            response_message = lang_initial_medical_q_es if self.is_spanish else lang_initial_medical_q_en
            self.call_data["medical_info_initial_ask_done"] = True
        elif self.call_data["medical_topics_to_cover"] and not self.call_data["medical_info_follow_up_done"]:
            thank_you_part = ""
            if newly_provided_topics_this_turn:
                natural_newly_provided = self._list_to_natural_language(newly_provided_topics_this_turn, current_topic_map, current_and_conjunction)
                thank_you_part = f"{lang_thank_you_for_info_es if self.is_spanish else lang_thank_you_for_info_en} {natural_newly_provided}. "

            follow_up_intro = lang_follow_up_intro_es if self.is_spanish else lang_follow_up_intro_en
            natural_missing_topics = self._list_to_natural_language(self.call_data["medical_topics_to_cover"], current_topic_map, current_or_conjunction)
            
            follow_up_query_base = lang_follow_up_query_es if self.is_spanish else lang_follow_up_query_en
            
            response_message = f"{thank_you_part}{follow_up_intro} {follow_up_query_base} {natural_missing_topics}?"
            self.call_data["medical_info_follow_up_done"] = True
        else:
            # All topics covered, or follow-up already done
            if newly_provided_topics_this_turn:
                natural_newly_provided = self._list_to_natural_language(newly_provided_topics_this_turn, current_topic_map, current_and_conjunction)
                thank_you_part = f"{lang_thank_you_for_info_es if self.is_spanish else lang_thank_you_for_info_en} {natural_newly_provided}. "
                anything_else_part = lang_anything_else_medical_es if self.is_spanish else lang_anything_else_medical_en
                response_message = f"{thank_you_part}{anything_else_part}"
            else:
                # Nothing new provided, and we're past follow-up or all covered.
                response_message = lang_generic_thanks_and_continue_es if self.is_spanish else lang_generic_thanks_and_continue_en
            
        await params.result_callback([{"role": "system", "content": response_message}])

    def _parse_requested_time(self, requested_date_time_str: Optional[str], requested_hour_24_format: Optional[int]) -> Optional[datetime]:
        """Parses LLM provided date/time string and hour into a datetime object."""
        if not requested_date_time_str or dateutil_parser is None:
            # If only hour is given, it implies today. However, LLM usually provides a string.
            # This part might need more sophisticated logic if only hour is common.
            if requested_hour_24_format is not None:
                now = datetime.now()
                try:
                    return now.replace(hour=requested_hour_24_format, minute=0, second=0, microsecond=0)
                except ValueError:
                    logger.warning(f"Invalid hour {requested_hour_24_format} for today.")
                    return None
            return None

        try:
            # Try parsing the string directly. dateutil is flexible.
            # We can provide a default time if the string is just a date.
            parsed_dt = dateutil_parser.parse(requested_date_time_str)
            
            # If hour is also provided, ensure it matches or use it as override if string is just date.
            if requested_hour_24_format is not None:
                if parsed_dt.hour != requested_hour_24_format or (parsed_dt.hour == 0 and parsed_dt.minute == 0 and requested_date_time_str.lower().endswith("midnight")):
                    # If the parsed string was just a date, it might default to midnight.
                    # Or if the hour is explicitly different.
                    try:
                        parsed_dt = parsed_dt.replace(hour=requested_hour_24_format, minute=0, second=0, microsecond=0) # Default to 00 minutes for the hour
                    except ValueError:
                        logger.warning(f"Could not apply hour {requested_hour_24_format} to parsed date {parsed_dt}. Returning original parse.")
            
            # Standardize to 00 or 30 minutes if not already set, aligning with typical slots
            if parsed_dt.minute not in [0, 30]:
                if parsed_dt.minute < 15:
                    parsed_dt = parsed_dt.replace(minute=0)
                elif parsed_dt.minute < 45:
                    parsed_dt = parsed_dt.replace(minute=30)
                else: # Round up to next hour
                    parsed_dt = (parsed_dt.replace(minute=0) + timedelta(hours=1))
            
            return parsed_dt
        except (dateutil_parser.ParserError, TypeError, ValueError) as e:
            logger.error(f"Could not parse date string '{requested_date_time_str}' with hour '{requested_hour_24_format}': {e}")
            return None

    def _is_slot_conflicting(self, requested_start_dt: datetime, appointments: List[Dict[str, Any]]) -> bool:
        """Checks if the requested slot conflicts with existing appointments."""
        if not requested_start_dt:
            return True # Cannot check a non-existent slot

        requested_end_dt = requested_start_dt + timedelta(minutes=APPOINTMENT_DURATION_MINUTES)

        for appt in appointments:
            if appt.get("status") == "confirmed":
                try:
                    existing_start_dt = dateutil_parser.isoparse(appt["scheduled_datetime_iso"])
                    existing_end_dt = existing_start_dt + timedelta(minutes=appt.get("duration_minutes", APPOINTMENT_DURATION_MINUTES))
                    
                    # Check for overlap
                    if max(requested_start_dt, existing_start_dt) < min(requested_end_dt, existing_end_dt):
                        logger.debug(f"Conflict: Requested {requested_start_dt}-{requested_end_dt} overlaps with existing {existing_start_dt}-{existing_end_dt}")
                        return True
                except Exception as e:
                    logger.error(f"Error parsing existing appointment datetime {appt.get('scheduled_datetime_iso')}: {e}")
                    # Treat as conflict if unsure
                    return True 
        return False

    def _find_next_available_slot(self, start_search_dt: datetime, appointments: List[Dict[str, Any]]) -> Optional[datetime]:
        """Finds the next available 30-minute slot starting from start_search_dt within business hours."""
        if dateutil_parser is None: return None # dateutil is required

        current_check_dt = start_search_dt

        # Align start_search_dt to the next 00 or 30 minute mark if it isn't already
        if current_check_dt.minute not in [0, 30]:
            if current_check_dt.minute < 30:
                current_check_dt = current_check_dt.replace(minute=30, second=0, microsecond=0)
            else:
                current_check_dt = (current_check_dt + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

        attempts = 0
        max_attempts = 48 * 5 # Search roughly 5 working days (48 half-hour slots per day)

        while attempts < max_attempts:
            attempts += 1
            # Check if current_check_dt is within business hours (Mon-Fri, 9 AM - 5 PM)
            is_weekday = current_check_dt.weekday() < 5  # 0=Mon, 4=Fri
            is_within_hours = BUSINESS_START_HOUR <= current_check_dt.hour < BUSINESS_END_HOUR 
                                # (slot must start before end hour)
            
            if is_weekday and is_within_hours:
                if not self._is_slot_conflicting(current_check_dt, appointments):
                    return current_check_dt
            
            # Advance to the next slot
            current_check_dt += timedelta(minutes=APPOINTMENT_DURATION_MINUTES)

            # If outside business hours, jump to next business day start
            if current_check_dt.hour >= BUSINESS_END_HOUR or not is_weekday:
                if current_check_dt.weekday() >= 4: # Friday or weekend, jump to next Monday
                    days_to_monday = (7 - current_check_dt.weekday()) % 7
                    if days_to_monday == 0 and current_check_dt.hour >= BUSINESS_START_HOUR : # If it is already Monday but past start, go to next day
                         days_to_monday = 1 if current_check_dt.weekday() != 4 else 3 # if sat/sun jump to mon, if fri jump to mon
                    if current_check_dt.weekday() == 5: days_to_monday = 2 # sat
                    if current_check_dt.weekday() == 6: days_to_monday = 1 # sun

                    current_check_dt = (current_check_dt + timedelta(days=days_to_monday)).replace(
                        hour=BUSINESS_START_HOUR, minute=0, second=0, microsecond=0)
                else: # Weekday, but past hours, jump to next day start
                    current_check_dt = (current_check_dt + timedelta(days=1)).replace(
                        hour=BUSINESS_START_HOUR, minute=0, second=0, microsecond=0)
        
        logger.warning("Could not find an available slot after extensive search.")
        return None

    async def end_call(self, params: FunctionCallParams):
        action = params.arguments.get("action")
        requested_date_time_str = params.arguments.get("requested_date_time")
        requested_hour_24_format = params.arguments.get("requested_hour_24_format")
        is_patient_confirming = params.arguments.get("is_appointment_confirmed_by_patient")
        final_appointment_details_str = params.arguments.get("final_appointment_details_for_goodbye")

        response_content = ""
        should_terminate_call_now = False

        # Load current appointments for checking/saving
        all_appointments = load_appointments()

        # Language strings (condensed for brevity, keep existing ones)
        lang_map = {
            "available": "is available" if not self.is_spanish else "está disponible",
            "unavailable": "is not available" if not self.is_spanish else "no está disponible",
            "confirm_q": "Would you like to confirm this appointment?" if not self.is_spanish else "¿Desea confirmar esta cita?",
            "how_about": "How about" if not self.is_spanish else "¿Qué tal",
            "another_time_q": "Is there another time you'd like to try, or perhaps a different day?" if not self.is_spanish else "¿Hay algún otro horario o día que le gustaría intentar?",
            "great_confirmed": "Great! Your appointment for" if not self.is_spanish else "¡Excelente! Su cita para",
            "is_confirmed": "is confirmed." if not self.is_spanish else "está confirmada.",
            "receive_text": "You'll receive a text message with the details shortly." if not self.is_spanish else "Recibirá un mensaje de texto con los detalles en breve.",
            "anything_else": "Is there anything else I can help with today?" if not self.is_spanish else "¿Hay algo más en lo que pueda ayudarle hoy?",
            "ok_not_scheduled": "Okay, we won't schedule that." if not self.is_spanish else "De acuerdo, no programaremos eso.",
            "try_different_time": "Would you like to try a different time, or is there something else?" if not self.is_spanish else "¿Le gustaría intentar un horario diferente, o hay algo más?",
            "thank_you_time": "Thank you for your time." if not self.is_spanish else "Gracias por su tiempo.",
            "look_forward_to_seeing_you": "We look forward to seeing you on" if not self.is_spanish else "Esperamos verle el",
            "wonderful_day": "Have a wonderful day!" if not self.is_spanish else "¡Que tenga un excelente día!",
            "need_specific_time": "I need a specific time to check. Could you please provide the date and time?" if not self.is_spanish else "Necesito un horario específico para verificar. ¿Podría proporcionar la fecha y la hora?",
            "unclear_request": "I'm not sure how to handle that request regarding appointments. Can you clarify?" if not self.is_spanish else "No estoy segura de cómo manejar esa solicitud. ¿Podría aclarar?",
            "could_not_parse_time": "I'm sorry, I had trouble understanding that time. Could you please try again, perhaps saying the full date and time?" if not self.is_spanish else "Lo siento, tuve problemas para entender esa hora. ¿Podría intentarlo de nuevo, quizás diciendo la fecha y hora completas?",
            "no_slots_found": "I'm sorry, I couldn't find any available slots soon. You might want to try specifying a different day or time range." if not self.is_spanish else "Lo siento, no pude encontrar ningún espacio disponible pronto. Quizás quiera intentar especificar un día o rango de tiempo diferente."
        }

        if action == "request_schedule":
            parsed_requested_dt = self._parse_requested_time(requested_date_time_str, requested_hour_24_format)

            if parsed_requested_dt:
                # Store the string version for potential confirmation display
                self.call_data["pending_appointment_request"] = {
                    "time_str": requested_date_time_str or parsed_requested_dt.strftime("%A, %B %d at %I:%M %p"), 
                    "datetime_iso": parsed_requested_dt.isoformat()
                }
                
                if not self._is_slot_conflicting(parsed_requested_dt, all_appointments):
                    response_content = f"{self.call_data['pending_appointment_request']['time_str']} {lang_map['available']}. {lang_map['confirm_q']}"
                else:
                    alternative_dt = self._find_next_available_slot(parsed_requested_dt, all_appointments)
                    if alternative_dt:
                        alt_time_str = alternative_dt.strftime("%A, %B %d at %I:%M %p") # User-friendly format
                        self.call_data["suggested_alternative_slot_details"] = {
                            "time_str": alt_time_str,
                            "datetime_iso": alternative_dt.isoformat()
                        }
                        response_content = f"Unfortunately, {self.call_data['pending_appointment_request']['time_str']} {lang_map['unavailable']}. {lang_map['how_about']} {alt_time_str}? {lang_map['another_time_q']}"
                    else:
                        response_content = lang_map['no_slots_found']
            else:
                response_content = lang_map['could_not_parse_time'] if requested_date_time_str else lang_map['need_specific_time']
        
        elif action == "confirm_schedule":
            slot_to_confirm_iso = None
            slot_to_confirm_display_str = None

            if self.call_data.get("suggested_alternative_slot_details") and is_patient_confirming is not False: # Patient might confirm suggestion or original
                 # If patient confirms AND a suggestion was made, assume they are confirming the suggestion.
                 # LLM should ideally clarify which one if ambiguous.
                confirmed_slot_details = self.call_data.pop("suggested_alternative_slot_details")
                slot_to_confirm_iso = confirmed_slot_details["datetime_iso"]
                slot_to_confirm_display_str = confirmed_slot_details["time_str"]
                # Clear pending if suggestion is acted upon
                self.call_data.pop("pending_appointment_request", None) 
            elif self.call_data.get("pending_appointment_request"):
                confirmed_slot_details = self.call_data.pop("pending_appointment_request")
                slot_to_confirm_iso = confirmed_slot_details["datetime_iso"]
                slot_to_confirm_display_str = confirmed_slot_details["time_str"]
            
            if is_patient_confirming and slot_to_confirm_iso and slot_to_confirm_display_str:
                try:
                    # Double check this slot isn't taken *again* just before saving (small race window)
                    # This is optional but safer if calls could be highly concurrent
                    # For now, assume the slot remains available.
                    
                    new_appointment = {
                        "appointment_id": str(uuid.uuid4()),
                        "scheduled_datetime_iso": slot_to_confirm_iso,
                        "duration_minutes": APPOINTMENT_DURATION_MINUTES,
                        "patient_name": self.call_data.get("patient_name", "Unknown"),
                        "intent": self.call_data.get("intent", "Not specified"),
                        "allergies": self.call_data.get("allergies", []),
                        "prescriptions": self.call_data.get("prescriptions", []),
                        "conditions": self.call_data.get("conditions", []),
                        "twilio_call_sid": self.twilio_call_sid,
                        "status": "confirmed"
                    }
                    all_appointments.append(new_appointment)
                    if save_appointments(all_appointments):
                        self.call_data["final_confirmed_appointment"] = slot_to_confirm_display_str
                        response_content = f"{lang_map['great_confirmed']} {slot_to_confirm_display_str} {lang_map['is_confirmed']} {lang_map['receive_text']} {lang_map['anything_else']}"
                    else:
                        response_content = "I apologize, there was an issue saving your appointment. Please try again or call back later." # Should be localized
                except Exception as e:
                    logger.error(f"Error creating appointment object or saving: {e}")
                    response_content = "I'm sorry, a system error occurred while confirming your appointment." # Should be localized
            
            elif not is_patient_confirming:
                response_content = f"{lang_map['ok_not_scheduled']} {lang_map['try_different_time']}"
                 # Clear any pending/suggested slots if declined
                self.call_data.pop("suggested_alternative_slot_details", None)
                self.call_data.pop("pending_appointment_request", None)
            else: # Should not happen: is_patient_confirming=True but no slot details found
                 response_content = f"{lang_map['ok_not_scheduled']} {lang_map['try_different_time']}"

        elif action == "terminate_interaction":
            final_appointment_to_mention = self.call_data.get("final_confirmed_appointment", final_appointment_details_str)
            if final_appointment_to_mention:
                response_content = f"{lang_map['thank_you_time']} <break time='1s'/> {lang_map['look_forward_to_seeing_you']} {final_appointment_to_mention}. {lang_map['wonderful_day']}"
            else:
                response_content = f"{lang_map['thank_you_time']} <break time='1s'/> {lang_map['wonderful_day']}"
            should_terminate_call_now = True
        
        else:
            response_content = lang_map['unclear_request']

        await params.result_callback([{"role": "system", "content": response_content}])

        if should_terminate_call_now:
            await asyncio.sleep(3) 
            logger.info("Terminating call by pushing EndTaskFrame.")
            # Check if pipeline is available to push EndTaskFrame
            if self.pipeline and hasattr(params.llm, 'push_frame'): # params.llm might not be the direct upstream to transport
                await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
            elif self.pipeline and hasattr(self.pipeline, 'push_frame'): # Try pushing to pipeline directly
                await self.pipeline.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
            else:
                logger.warning("Could not push EndTaskFrame: No suitable push_frame method found on llm or pipeline.")

    async def save_data(self, args, result_callback):
        try:
            current_time = datetime.now()
            date_str = current_time.strftime("%Y%m%d_%H%M%S")
            
            # Use twilio_call_sid for the filename
            if not self.twilio_call_sid:
                logger.error("twilio_call_sid is not set in IntakeProcessor. Cannot save data with proper filename.")
                # Fallback filename if twilio_call_sid is missing for some reason
                filename = f"unknown_call_{date_str}.json"
            else:
                filename = f"call_log_{self.twilio_call_sid}_{date_str}.json"

            data_dir = os.path.join(script_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            filepath = os.path.join(data_dir, filename)

            # Consolidate all data for saving
            # Prioritize self.call_data, then update with any specific status from args
            final_data_to_save = {
                "call_sid": self.twilio_call_sid,
                "start_time_iso": self.call_data.get("call_start_time_iso", current_time.isoformat()), # Assuming you might add this elsewhere
                "end_time_iso": current_time.isoformat(),
                **self.call_data, # All collected call-specific data
            }

            # Update with any status passed in args (e.g., from on_participant_left)
            if args and isinstance(args, dict) and "call_status" in args:
                final_data_to_save["call_status"] = args.get("call_status")
            else:
                final_data_to_save.setdefault("call_status", "completed_unknown_reason")

            # Add conversation history
            if self.context and hasattr(self.context, 'messages') and self.context.messages:
                final_data_to_save["conversation_history"] = self.context.messages
            else:
                final_data_to_save["conversation_history"] = []
                logger.warning(f"No conversation history found in context for {self.twilio_call_sid}")

            with open(filepath, "w") as f:
                json.dump(final_data_to_save, f, indent=2)
            logger.info(f"Saved call data for {self.twilio_call_sid} to {filepath}")
                
        except Exception as e:
            logger.error(f"Error saving data for call {self.twilio_call_sid if hasattr(self, 'twilio_call_sid') else 'UNKNOWN'}: {e}")
        
        # Original save_data didn't seem to use result_callback, so keeping it None or logging if it were present
        if result_callback:
            logger.debug("save_data was called with a result_callback, but it is not used in this implementation.")


async def main(room_url: str, token: str, twilio_call_sid: str, daily_room_sip_uri_for_bot_dialin_config: str):
    logger.info(f"Bot starting with Room URL: {room_url}")
    logger.info(f"Bot using Twilio CallSid: {twilio_call_sid}")
    logger.info(f"Bot received Daily Room SIP URI for dial-in config: {daily_room_sip_uri_for_bot_dialin_config}")

    # Initialize Twilio client
    # Ensure TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN are in environment variables
    twilio_client = None
    twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")

    if twilio_account_sid and twilio_auth_token:
        twilio_client = Client(twilio_account_sid, twilio_auth_token)
        logger.info("Twilio client initialized successfully.")
    else:
        logger.error("TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN not found in environment. Twilio client not initialized.")
        # Depending on the desired behavior, you might want to exit or handle this differently.
        # For now, the bot will continue but won't be able to update the Twilio call.

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
                vad_analyzer=SileroVADAnalyzer(),
                transcription_enabled=True,
                # Enable dialin_settings if you expect this bot to be dialed into via SIP from Twilio
                # For the on_dialin_ready event to trigger as intended for call forwarding.
                dialin_settings={ 'url': daily_room_sip_uri_for_bot_dialin_config } if daily_room_sip_uri_for_bot_dialin_config else None 
            )
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # English-speaking Lady
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        messages = [] # Keep initial messages empty, system prompt is in IntakeProcessor
        context = OpenAILLMContext(messages=messages) # Tools are set in IntakeProcessor
        
        intake = IntakeProcessor(context, twilio_call_sid) # Pass twilio_call_sid here
        # Pass transport to intake processor for graceful shutdown
        context_aggregator = llm.create_context_aggregator(context)

        # Register functions with the LLM
        llm.register_function("classify_intent", intake.classify_intent)
        llm.register_function("collect_medical_info", intake.collect_medical_info)
        llm.register_function("end_call", intake.end_call)

        fl = FrameLogger("LLM Output")

        pipeline = Pipeline(
            [
                transport.input(),
                context_aggregator.user(),
                llm,
                fl,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )
        intake.pipeline = pipeline # For potential use within intake, though transport_service is preferred for leave()

        task = PipelineTask(pipeline, params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ))
        # intake.pipeline.task = task # Not typically needed this way

        # Event handler for Twilio call forwarding
        @transport.event_handler("on_dialin_ready")
        async def on_dialin_ready(transport_ref, actual_sip_uri_of_daily_room: str):
            logger.info(f"Daily dial-in ready. SIP URI for this Daily session: {actual_sip_uri_of_daily_room}")

            if not twilio_client:
                logger.error("Twilio client not initialized. Cannot update Twilio call to bridge to Daily SIP.")
                return
            
            if not twilio_call_sid:
                logger.error("Twilio CallSid is not available. Cannot update Twilio call.")
                return
            
            if not actual_sip_uri_of_daily_room:
                logger.error("Actual SIP URI of Daily room is not available. Cannot update Twilio call.")
                return

            try:
                logger.info(f"Attempting to update Twilio call {twilio_call_sid} to connect to Daily SIP URI: {actual_sip_uri_of_daily_room}")
                # Construct TwiML to dial the Daily room's SIP URI
                # Note: Twilio requires the sip: prefix. The actual_sip_uri_of_daily_room might already have it.
                # Ensure it does, or add if missing.
                sip_target_for_twilio = actual_sip_uri_of_daily_room
                if not sip_target_for_twilio.startswith("sip:"):
                    sip_target_for_twilio = f"sip:{actual_sip_uri_of_daily_room}"
                
                # Create TwiML to <Dial><Sip> the Daily room
                twiml_response_for_update = f'''<Response><Dial><Sip>{sip_target_for_twilio}</Sip></Dial></Response>'''

                call = twilio_client.calls(twilio_call_sid).update(
                    twiml=twiml_response_for_update
                )
                logger.info(f"Twilio call {twilio_call_sid} update initiated. Status: {call.status}. New TwiML will dial {sip_target_for_twilio}")
            except Exception as e:
                logger.error(f"Error updating Twilio call {twilio_call_sid}: {e}")
                # Potentially push an error frame or end the task if this fails critically

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport_ref, participant): # Renamed to transport_ref to avoid clash
            await transport_ref.capture_participant_transcription(participant["id"])
            # Let LLM initiate conversation based on system prompt in IntakeProcessor's context
            await task.queue_frames([OpenAILLMContextFrame(context)])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport_ref, participant, *args):
            logger.info(f"Participant {participant.get('id', 'Unknown')} left. Saving data and cancelling task.")
            await intake.save_data({"call_status": "participant_left"}, None)
            logger.info(f"Attempting to cancel pipeline task for participant {participant.get('id', 'Unknown')}.")
            await task.cancel() # Directly attempt to cancel the task.
            logger.info(f"Pipeline task cancellation requested for participant {participant.get('id', 'Unknown')}.")

        runner = PipelineRunner()
        try:
            await runner.run(task)
        finally:
            logger.info("Pipeline task finished or runner exited.")
            # Transport cleanup should be handled by the runner or Daily SDK when connection drops
            # or when the task is properly cancelled/completed.
            # Explicitly closing transport here can be problematic if Daily is already cleaning up.
            # if transport and not transport.is_closed:
            #     logger.info("Ensuring transport is closed in main finally block.")
            #     await transport.stop() # Or appropriate close/disconnect method if available and needed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Daily Bot")
    parser.add_argument("-u", "--url", type=str, help="Daily room URL", required=True)
    parser.add_argument("-t", "--token", type=str, help="Daily token", required=True)
    parser.add_argument("-cid", "--call_id", type=str, help="Twilio CallSid", required=True) # Changed from -i to -cid for clarity
    parser.add_argument("-s", "--sip_uri", type=str, help="Daily room SIP URI (for bot to configure Daily dial-in)", required=True)
    config = parser.parse_args()

    asyncio.run(main(config.url, config.token, config.call_id, config.sip_uri))