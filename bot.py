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
from src.scheduling_helpers import parse_requested_time, is_slot_conflicting, find_next_available_slot, is_slot_within_business_hours

load_dotenv(override=True)

# Define the appointment duration in minutes
APPOINTMENT_DURATION_MINUTES = 45

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
    def __init__(self, context: OpenAILLMContext, twilio_call_sid: str):
        logger.debug(f"[DEBUG] IntakeProcessor __init__ for CallSid: {twilio_call_sid}")
        self.twilio_call_sid = twilio_call_sid
        self.call_data = {}
        self.is_spanish = False
        self.context = context
        self.pipeline = None
        self.call_data["patient_name"] = None
        self.call_data["intent"] = None
        self.call_data["medical_info_initial_ask_done"] = False
        self.call_data["medical_info_follow_up_done"] = False
        self.call_data["medical_topics_to_cover"] = ["allergies", "conditions", "prescriptions"]
        context.add_message(
            {
                "role": "system",
                "content": """You are Jessica, an agent for Bay Area Health. Your job is to collect important information and assist with appointment scheduling. Be polite, professional, and keep responses natural. You are not a medical professional.

Important conversation guidelines:
1. Start by introducing yourself and asking for the patient's name. Your first function call MUST be to 'classify_intent' with the patient's name and their initial reason for calling (if provided).
2. When calling 'classify_intent', if the patient mentions scheduling an appointment, try to also capture the specific reason for the visit  in the 'appointment_reason' parameter.
3. Wait for the user to finish speaking before responding. Use <break time='1s'/> for noticeable pauses.
4. Let the patient guide the conversation.
5. After an appointment is successfully booked via 'end_call' (action: confirm_schedule), YOUR NEXT STEP IS TO ASK ABOUT MEDICAL INFORMATION. You should then call 'collect_medical_info'. If the user has already provided some medical details during scheduling, 'collect_medical_info' will handle follow-ups. If not, it will ask the initial medical question. Do this BEFORE considering ending the call.
6. If the patient mentions medications, allergies, or conditions at any other time, use 'collect_medical_info'.
7. For appointment requests:
    - Use the 'end_call' function with action 'request_schedule'. Provide 'requested_date_time' (e.g., "tomorrow 3 PM") AND 'requested_hour_24_format' (e.g., 15 for 3 PM).
    - If a proposed time is confirmed by the patient, call 'end_call' with action 'confirm_schedule' and 'is_appointment_confirmed_by_patient': true. Include the confirmed 'requested_date_time' and 'requested_hour_24_format'.
    - If a proposed time is declined, or the patient wants a different time, call 'end_call' with action 'request_schedule' and the new 'requested_date_time' and 'requested_hour_24_format'.
    - If the patient wants to stop scheduling after a decline, call 'end_call' with action 'terminate_interaction'.
8. When the conversation is truly over (all information gathered, including medical info post-scheduling, and no further questions), call 'end_call' with action 'terminate_interaction'. If an appointment was booked, provide 'final_appointment_details_for_goodbye'.
9. If Spanish is detected, switch to Spanish for all interactions.
""",
            }
        )
        context.set_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "classify_intent",
                        "description": "Classify the initial intent of the patient's visit AFTER they provide their name and initial reason for calling. This function MUST be called early in the conversation.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "patient_name": {"type": "string", "description": "The patient's full name as stated by them."},
                                "intent": {"type": "string", "description": "Broad classified intent of the call (e.g., appointment_scheduling, billing_inquiry, general_info)."},
                                "appointment_reason": {"type": "string", "description": "If intent is 'appointment_scheduling', the specific medical reason for the visit (e.g., 'annual check-up', 'flu symptoms', 'follow-up')."},
                                "details": {"type": "string", "description": "Other relevant details about the call intent or reason for visit not covered elsewhere."},
                                "is_spanish": {"type": "boolean", "description": "Is the user speaking Spanish?"}
                            },
                            "required": ["patient_name", "intent"]
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "collect_medical_info",
                        "description": "Collect or inquire about medical information (prescriptions, allergies, conditions). Call this if the patient mentions these, or to proactively ask once if appropriate (e.g., after scheduling is done or if they bring up a health topic).",
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
                        "description": "Handles appointment scheduling logic (requests, availability checks, confirmations) or concludes the call. This function is used for multiple scheduling steps.",
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
                                    "description": "The full date and time string for an appointment request (e.g., 'tomorrow 3 PM', 'next Tuesday at 10 AM') or confirmation. Used with 'request_schedule' and 'confirm_schedule'."
                                },
                                "requested_hour_24_format": {
                                    "type": "integer",
                                    "description": "The hour (0-23) of the 'requested_date_time'. Essential for availability check and confirmation. Used with 'request_schedule' and 'confirm_schedule'."
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
        logger.debug(f"[DEBUG] classify_intent ENTRY: self.call_data before update: {json.dumps(self.call_data, indent=2)}")
        logger.debug(f"[DEBUG] classify_intent ARGS: {json.dumps(params.arguments, indent=2)}")
        
        if "patient_name" in params.arguments:
            self.call_data["patient_name"] = params.arguments["patient_name"]
        if "intent" in params.arguments:
            self.call_data["intent"] = params.arguments["intent"]
        if "appointment_reason" in params.arguments:
            self.call_data["appointment_reason"] = params.arguments["appointment_reason"]
        if "details" in params.arguments:
            self.call_data["details"] = params.arguments["details"]
        if "is_spanish" in params.arguments:
            self.is_spanish = params.arguments.get("is_spanish", False)
        
        patient_name_to_use = self.call_data.get("patient_name", "there")
        primary_reason_for_call = self.call_data.get("appointment_reason", self.call_data.get("intent"))
        
        response_message = ""

        if not primary_reason_for_call:
            if self.is_spanish:
                response_message = f"Gracias, {patient_name_to_use}. <break time='1s'/> ¿Y cuál es el motivo de su llamada hoy?"
            else:
                response_message = f"Thanks, {patient_name_to_use}. <break time='1s'/> And what is the reason for your call today?"
        else:
            display_reason = primary_reason_for_call or (self.call_data.get("intent", "su consulta") if self.is_spanish else self.call_data.get("intent", "your inquiry"))
            if self.is_spanish:
                response_message = f"Entendido, {patient_name_to_use}. Usted se contactó por {display_reason}. <break time='1s'/> ¿Hay algo más en lo que pueda ayudarle o alguna otra información que desee compartir?"
            else:
                response_message = f"Okay, {patient_name_to_use}. I understand you're calling about {display_reason}. <break time='1s'/> Is there anything else I can help you with, or any other information you'd like to share?"
            
            # The block for narrowing tools has been removed.
            # The LLM will continue to have access to all tools defined in __init__.
        
        logger.debug(f"[DEBUG] classify_intent EXIT: self.call_data after update: {json.dumps(self.call_data, indent=2)}")
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
        logger.debug(f"[DEBUG] collect_medical_info ENTRY: self.call_data before update: {json.dumps(self.call_data, indent=2)}")
        logger.debug(f"[DEBUG] collect_medical_info ARGS: {json.dumps(params.arguments, indent=2)}")
        self.call_data.update(params.arguments)
        for topic in ["allergies", "prescriptions", "conditions"]:
            if topic in params.arguments and isinstance(params.arguments[topic], list):
                self.call_data[topic] = params.arguments[topic]
            elif topic not in self.call_data:
                self.call_data[topic] = [] 

        # Check if we need to update a recently confirmed appointment in appointments.json
        confirmed_appointment_id_to_update = self.call_data.get("confirmed_appointment_id")
        if confirmed_appointment_id_to_update:
            logger.info(f"[DEBUG] Medical info collected, attempting to update appointment ID: {confirmed_appointment_id_to_update} in appointments.json")
            all_appointments = load_appointments()
            updated = False
            for i, appt in enumerate(all_appointments):
                if appt.get("appointment_id") == confirmed_appointment_id_to_update:
                    logger.debug(f"[DEBUG] Found appointment {confirmed_appointment_id_to_update} to update with medical info.")
                    all_appointments[i]["allergies"] = self.call_data.get("allergies", [])
                    all_appointments[i]["prescriptions"] = self.call_data.get("prescriptions", [])
                    all_appointments[i]["conditions"] = self.call_data.get("conditions", [])
                    if save_appointments(all_appointments):
                        logger.info(f"[DEBUG] Successfully updated appointment {confirmed_appointment_id_to_update} in appointments.json with medical info.")
                        # Optionally clear the ID to prevent re-updates if collect_medical_info is called again for some reason
                        # self.call_data.pop("confirmed_appointment_id", None) 
                    else:
                        logger.error(f"[DEBUG] Failed to save updated medical info for appointment {confirmed_appointment_id_to_update} to appointments.json.")
                    updated = True
                    break
            if not updated:
                logger.warning(f"[DEBUG] Could not find appointment ID {confirmed_appointment_id_to_update} in appointments.json to update medical info.")

        if "medical_topics_to_cover" not in self.call_data:
            self.call_data["medical_topics_to_cover"] = ["allergies", "conditions", "prescriptions"]
        newly_provided_topics_this_turn = []
        topics_to_check = ["allergies", "conditions", "prescriptions"]
        for topic in topics_to_check:
            if params.arguments.get(topic) and isinstance(params.arguments.get(topic), list) and len(params.arguments.get(topic)) > 0:
                newly_provided_topics_this_turn.append(topic)
                if topic in self.call_data["medical_topics_to_cover"]:
                    self.call_data["medical_topics_to_cover"].remove(topic)
        
        response_message = ""
        topic_map_es = {"allergies": "alergias", "conditions": "condiciones médicas", "prescriptions": "prescripciones"}
        topic_map_en = {"allergies": "allergies", "conditions": "medical conditions", "prescriptions": "prescriptions"}
        lang_and_es = "y"
        lang_and_en = "and"
        lang_or_es = "o"
        lang_or_en = "or"
        current_topic_map = topic_map_es if self.is_spanish else topic_map_en
        current_and_conjunction = lang_and_es if self.is_spanish else lang_and_en
        current_or_conjunction = lang_or_es if self.is_spanish else lang_or_en
        lang_initial_medical_q_es = "¿Para asegurarnos de tener todos los detalles, podría informarme sobre cualquier prescripción actual, alergia o condición médica preexistente que el médico deba conocer?"
        lang_initial_medical_q_en = "To ensure we have all details, could you tell me about any current prescriptions, allergies, or existing medical conditions the doctor should be aware of?"
        lang_thank_you_for_info_es = "Gracias por la información sobre sus"
        lang_thank_you_for_info_en = "Thanks for the information about your"
        lang_follow_up_intro_es = "Solo para estar seguros,"
        lang_follow_up_intro_en = "Just to be sure,"
        lang_follow_up_query_es = "¿podría también informarme sobre"
        lang_follow_up_query_en = "could you also tell me about any"
        lang_anything_else_medical_es = "¿Hay algo más que desee agregar sobre estos temas médicos?"
        lang_anything_else_medical_en = "Is there anything else you'd like to add on these medical topics?"
        lang_generic_thanks_and_continue_es = "Entendido, gracias. ¿Hay algo más en lo que pueda ayudarle?"
        lang_generic_thanks_and_continue_en = "Okay, thank you. Is there anything else I can help with?"

        if not self.call_data.get("medical_info_initial_ask_done", False):
            response_message = lang_initial_medical_q_es if self.is_spanish else lang_initial_medical_q_en
            self.call_data["medical_info_initial_ask_done"] = True
        elif self.call_data.get("medical_topics_to_cover") and not self.call_data.get("medical_info_follow_up_done", False):
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
            if newly_provided_topics_this_turn:
                natural_newly_provided = self._list_to_natural_language(newly_provided_topics_this_turn, current_topic_map, current_and_conjunction)
                thank_you_part = f"{lang_thank_you_for_info_es if self.is_spanish else lang_thank_you_for_info_en} {natural_newly_provided}. "
                anything_else_part = lang_anything_else_medical_es if self.is_spanish else lang_anything_else_medical_en
                response_message = f"{thank_you_part}{anything_else_part}"
            else:
                response_message = lang_generic_thanks_and_continue_es if self.is_spanish else lang_generic_thanks_and_continue_en

        logger.debug(f"[DEBUG] collect_medical_info EXIT: self.call_data after update: {json.dumps(self.call_data, indent=2)}")
        await params.result_callback([{"role": "system", "content": response_message}])

    async def end_call(self, params: FunctionCallParams):
        action = params.arguments.get("action")
        requested_date_time_str = params.arguments.get("requested_date_time")
        requested_hour_24_format = params.arguments.get("requested_hour_24_format")
        is_patient_confirming = params.arguments.get("is_appointment_confirmed_by_patient")
        final_appointment_details_str = params.arguments.get("final_appointment_details_for_goodbye")

        response_content = ""
        should_terminate_call_now = False
        all_appointments = load_appointments()
        
        lang_map = { 
            "available": "is available" if not self.is_spanish else "está disponible",
            "unavailable": "is not available" if not self.is_spanish else "no está disponible",
            "confirm_q": "Would you like to confirm this appointment?" if not self.is_spanish else "¿Desea confirmar esta cita?",
            "how_about": "How about" if not self.is_spanish else "¿Qué tal",
            "another_time_q": "Is there another time you'd like to try, or perhaps a different day?" if not self.is_spanish else "¿Hay algún otro horario o día que le gustaría intentar?",
            "great_confirmed": "Great! Your appointment for" if not self.is_spanish else "¡Excelente! Su cita para",
            "is_confirmed": "is confirmed." if not self.is_spanish else "está confirmada.",
            "receive_text": "You'll receive a text message with the details shortly." if not self.is_spanish else "Recibirá un mensaje de texto con los detalles en breve.",
            "post_confirm_medical_prompt_en": "To help the doctor prepare, could you tell me about any current prescriptions, allergies, or existing medical conditions the doctor should be aware of?",
            "post_confirm_medical_prompt_es": "Para ayudar al médico a prepararse, ¿podría informarme sobre cualquier prescripción actual, alergia o condición médica preexistente que el médico deba conocer?",
            "anything_else": "Is there anything else I can help with today?" if not self.is_spanish else "¿Hay algo más en lo que pueda ayudarle hoy?",
            "ok_not_scheduled": "Okay, we won't schedule that." if not self.is_spanish else "De acuerdo, no programaremos eso.",
            "try_different_time": "Would you like to try a different time, or is there something else?" if not self.is_spanish else "¿Le gustaría intentar un horario diferente, o hay algo más?",
            "thank_you_time": "Thank you for your time." if not self.is_spanish else "Gracias por su tiempo.",
            "look_forward_to_seeing_you": "We look forward to seeing you on" if not self.is_spanish else "Esperamos verle el",
            "wonderful_day": "Have a wonderful day!" if not self.is_spanish else "¡Que tenga un excelente día!",
            "need_specific_time": "I need a specific time to check. Could you please provide the date and time?" if not self.is_spanish else "Necesito un horario específico para verificar. ¿Podría proporcionar la fecha y la hora?",
            "unclear_request": "I'm not sure how to handle that request regarding appointments. Can you clarify?" if not self.is_spanish else "No estoy segura de cómo manejar esa solicitud. ¿Podría aclarar?",
            "could_not_parse_time": "I'm sorry, I had trouble understanding that time. Could you please try again, perhaps saying the full date and time?" if not self.is_spanish else "Lo siento, tuve problemas para entender esa hora. ¿Podría intentarlo de nuevo, quizás diciendo la fecha y hora completas?",
            "no_slots_found": "I'm sorry, I couldn't find any available slots soon. You might want to try specifying a different day or time range." if not self.is_spanish else "Lo siento, no pude encontrar ningún espacio disponible pronto. Quizás quiera intentar especificar un día o rango de tiempo diferente.",
            "outside_business_hours": "Our clinic is open on weekdays from 9 AM to 5 PM. Please select a time within our business hours." if not self.is_spanish else "Nuestra clínica está abierta de lunes a viernes de 9 AM a 5 PM. Por favor, seleccione un horario dentro de nuestro horario de atención."
        }

        if action == "request_schedule":
            parsed_requested_dt = parse_requested_time(requested_date_time_str, requested_hour_24_format)
            if parsed_requested_dt:
                if not is_slot_within_business_hours(parsed_requested_dt):
                    response_content = lang_map['outside_business_hours']
                else:
                    self.call_data["pending_appointment_request"] = {
                        "time_str": requested_date_time_str or parsed_requested_dt.strftime("%A, %B %d at %I:%M %p"),
                        "datetime_iso": parsed_requested_dt.isoformat()
                    }
                    logger.debug(f"[DEBUG] end_call (request_schedule): Updated pending_appointment_request in self.call_data: {json.dumps(self.call_data.get('pending_appointment_request'), indent=2)}")
                    if not is_slot_conflicting(parsed_requested_dt, all_appointments):
                        response_content = f"{self.call_data['pending_appointment_request']['time_str']} {lang_map['available']}. {lang_map['confirm_q']}"
                    else:
                        alternative_dt = find_next_available_slot(parsed_requested_dt, all_appointments)
                        if alternative_dt:
                            alt_time_str = alternative_dt.strftime("%A, %B %d at %I:%M %p")
                            self.call_data["suggested_alternative_slot_details"] = {
                                "time_str": alt_time_str,
                                "datetime_iso": alternative_dt.isoformat()
                            }
                            logger.debug(f"[DEBUG] end_call (request_schedule): Updated suggested_alternative_slot_details in self.call_data: {json.dumps(self.call_data.get('suggested_alternative_slot_details'), indent=2)}")
                            response_content = f"Unfortunately, {self.call_data['pending_appointment_request']['time_str']} {lang_map['unavailable']}. {lang_map['how_about']} {alt_time_str}? {lang_map['another_time_q']}"
                        else:
                            response_content = lang_map['no_slots_found']
            else:
                response_content = lang_map['could_not_parse_time'] if requested_date_time_str else lang_map['need_specific_time']

        elif action == "confirm_schedule":
            logger.debug(f"[DEBUG] end_call (confirm_schedule ENTRY): self.call_data: {json.dumps(self.call_data, indent=2)}")
            logger.debug(f"[DEBUG] end_call (confirm_schedule ENTRY): is_patient_confirming: {is_patient_confirming}")
            slot_to_confirm_iso = None
            slot_to_confirm_display_str = None

            if self.call_data.get("suggested_alternative_slot_details") and is_patient_confirming:
                logger.debug("[DEBUG] Confirming based on suggested_alternative_slot_details.")
                confirmed_slot_details = self.call_data.pop("suggested_alternative_slot_details")
                slot_to_confirm_iso = confirmed_slot_details["datetime_iso"]
                slot_to_confirm_display_str = confirmed_slot_details["time_str"]
                self.call_data.pop("pending_appointment_request", None) 
            elif self.call_data.get("pending_appointment_request"):
                logger.debug("[DEBUG] Confirming based on pending_appointment_request.")
                confirmed_slot_details = self.call_data.pop("pending_appointment_request")
                slot_to_confirm_iso = confirmed_slot_details["datetime_iso"]
                slot_to_confirm_display_str = confirmed_slot_details["time_str"]
            else:
                logger.warning("[DEBUG] No pending or suggested slot details found in call_data for confirmation.")
            
            logger.debug(f"[DEBUG] Slot details for confirmation: ISO='{slot_to_confirm_iso}', Display='{slot_to_confirm_display_str}'")

            if is_patient_confirming and slot_to_confirm_iso and slot_to_confirm_display_str:
                logger.debug(f"[DEBUG] end_call (confirm_schedule): Current self.call_data before creating new_appointment dict: {json.dumps(self.call_data, indent=2)}")
                try:
                    appointment_intent_for_record = self.call_data.get("appointment_reason") or self.call_data.get("intent", "Not specified")
                    new_appointment_id = str(uuid.uuid4())
                    new_appointment = {
                        "appointment_id": new_appointment_id, # Store the new ID
                        "scheduled_datetime_iso": slot_to_confirm_iso,
                        "duration_minutes": APPOINTMENT_DURATION_MINUTES,
                        "patient_name": self.call_data.get("patient_name", "Unknown"),
                        "intent": appointment_intent_for_record,
                        "allergies": [], # Initially save with empty medical info
                        "prescriptions": [],
                        "conditions": [],
                        "twilio_call_sid": self.twilio_call_sid,
                        "status": "confirmed"
                    }
                    all_appointments = load_appointments() # Load fresh list
                    all_appointments.append(new_appointment)
                    if save_appointments(all_appointments):
                        self.call_data["final_confirmed_appointment"] = slot_to_confirm_display_str
                        self.call_data["confirmed_appointment_id"] = new_appointment_id # Store ID for potential update
                        logger.info(f"Successfully saved initial appointment: {new_appointment_id}")
                        
                        confirmation_message = f"{lang_map['great_confirmed']} {slot_to_confirm_display_str} {lang_map['is_confirmed']} {lang_map['receive_text']}. "
                        medical_prompt = lang_map['post_confirm_medical_prompt_es'] if self.is_spanish else lang_map['post_confirm_medical_prompt_en']
                        response_content = confirmation_message + medical_prompt
                        self.call_data["medical_info_initial_ask_done"] = True 
                        self.call_data["medical_topics_to_cover"] = ["allergies", "conditions", "prescriptions"]
                        self.call_data["medical_info_follow_up_done"] = False
                    else:
                        response_content = "I apologize, there was an issue saving your appointment. Please try again or call back later."
                        logger.error(f"Failed to save appointments file after attempting to add appointment for {slot_to_confirm_display_str}.")
                except Exception as e:
                    logger.error(f"Error creating appointment object or saving: {e}")
                    response_content = "I'm sorry, a system error occurred while confirming your appointment."
            elif not is_patient_confirming:
                logger.debug("[DEBUG] Patient did not confirm. Clearing pending/suggested slots.")
                response_content = f"{lang_map['ok_not_scheduled']} {lang_map['try_different_time']}"
                self.call_data.pop("suggested_alternative_slot_details", None)
                self.call_data.pop("pending_appointment_request", None)
            else:
                 logger.warning(f"[DEBUG] Condition for creating appointment not met. is_patient_confirming: {is_patient_confirming}, slot_iso: {slot_to_confirm_iso}, slot_display: {slot_to_confirm_display_str}")
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
            if self.pipeline and hasattr(params.llm, 'push_frame'):
                await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
            elif self.pipeline and hasattr(self.pipeline, 'push_frame'):
                await self.pipeline.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
            else:
                logger.warning("Could not push EndTaskFrame: No suitable push_frame method found on llm or pipeline.")

    async def save_data(self, args, result_callback):
        logger.debug(f"[DEBUG] save_data called. Args: {args}")
        if not hasattr(self, 'twilio_call_sid') or not self.twilio_call_sid:
            logger.error("[DEBUG] twilio_call_sid is not set in IntakeProcessor. Individual call log may use fallback name or fail if twilio_call_sid is essential later.")
            # Ensure self.twilio_call_sid exists for filename generation even if it's a placeholder
            effective_call_sid = f"unknown_sid_at_save_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        else:
            effective_call_sid = self.twilio_call_sid
            logger.debug(f"[DEBUG] save_data: twilio_call_sid is '{effective_call_sid}'")

        try:
            current_time = datetime.now()
            date_str = current_time.strftime("%Y%m%d_%H%M%S")
            
            filename = f"call_log_{effective_call_sid}_{date_str}.json"

            data_dir = os.path.join(script_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            filepath = os.path.join(data_dir, filename)

            final_data_to_save = {
                "call_sid": effective_call_sid,
                "start_time_iso": self.call_data.get("call_start_time_iso", current_time.isoformat()),
                "end_time_iso": current_time.isoformat(),
                **self.call_data,
            }

            if args and isinstance(args, dict) and "call_status" in args:
                final_data_to_save["call_status"] = args.get("call_status")
            else:
                final_data_to_save.setdefault("call_status", "completed_unknown_reason")

            # For conversation history, only include serializable parts like the message content
            serializable_history = []
            if self.context and hasattr(self.context, 'messages') and self.context.messages:
                for msg in self.context.messages:
                    # Ensure msg is a dict and only take serializable parts, typically role and content
                    if isinstance(msg, dict):
                        serializable_history.append({
                            "role": msg.get("role"),
                            "content": msg.get("content")
                        })
                    # If context stores messages in another format, adjust accordingly
            final_data_to_save["conversation_history"] = serializable_history
            
            logger.debug(f"[DEBUG] save_data: Attempting to save to {filepath}. Data keys: {list(final_data_to_save.keys())}")

            with open(filepath, "w") as f:
                json.dump(final_data_to_save, f, indent=2)
            logger.info(f"Saved individual call data for {effective_call_sid} to {filepath}")
                
        except Exception as e:
            logger.error(f"[DEBUG] Error in save_data for call {effective_call_sid}: {e}", exc_info=True)
        
        if result_callback:
            logger.debug("save_data was called with a result_callback, but it is not used in this implementation.")


async def main(room_url: str, token: str, twilio_call_sid: str, daily_room_sip_uri_for_bot_dialin_config: str):
    logger.info(f"[DEBUG] bot.py main started for CallSid: {twilio_call_sid}")
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
            participant_id = participant.get('id', 'Unknown')
            logger.info(f"[DEBUG] on_participant_left triggered for participant: {participant_id}. CallSid: {twilio_call_sid}")
            try:
                # Pass a copy of call_data to avoid issues if it's modified elsewhere during async operations
                await intake.save_data({"call_status": "participant_left", **intake.call_data.copy()}, None)
                logger.info(f"[DEBUG] on_participant_left: save_data call initiated for participant {participant_id}.")
            except Exception as e_save:
                logger.error(f"[DEBUG] on_participant_left: Error calling save_data for {participant_id}: {e_save}", exc_info=True)
            
            logger.info(f"Attempting to cancel pipeline task for participant {participant_id}.")
            try:
                await task.cancel()
                logger.info(f"Pipeline task cancellation requested for participant {participant_id}.")
            except Exception as e_cancel:
                logger.error(f"[DEBUG] on_participant_left: Error cancelling task for {participant_id}: {e_cancel}", exc_info=True)

        runner = PipelineRunner()
        try:
            await runner.run(task)
        finally:
            logger.info(f"Pipeline task finished or runner exited for CallSid: {twilio_call_sid}.")
            # Explicitly call save_data here as a fallback if on_participant_left didn't cover all exit paths
            # This ensures data is saved even if the call ends through other means (e.g. runner error, task completion without explicit left event)
            logger.info(f"[DEBUG] Fallback save_data in main finally block for CallSid: {twilio_call_sid}")
            try:
                await intake.save_data({"call_status": "completed_in_main_finally", **intake.call_data.copy()}, None)
                logger.info(f"[DEBUG] Fallback save_data call completed in main finally for CallSid: {twilio_call_sid}.")
            except Exception as e_final_save:
                logger.error(f"[DEBUG] Error in fallback save_data for CallSid {twilio_call_sid}: {e_final_save}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Daily Bot")
    parser.add_argument("-u", "--url", type=str, help="Daily room URL", required=True)
    parser.add_argument("-t", "--token", type=str, help="Daily token", required=True)
    parser.add_argument("-cid", "--call_id", type=str, help="Twilio CallSid", required=True) # Changed from -i to -cid for clarity
    parser.add_argument("-s", "--sip_uri", type=str, help="Daily room SIP URI (for bot to configure Daily dial-in)", required=True)
    config = parser.parse_args()

    # Define a simple retry mechanism
    retry_count = 0
    max_retries = 2
    while retry_count <= max_retries:
        try:
            asyncio.run(main(config.url, config.token, config.call_id, config.sip_uri))
            break  # Exit loop if main completes successfully
        except Exception as e:
            # This is a broad catch. For production, you might want to be more specific.
            # For instance, catch httpx.RemoteProtocolError specifically if that's the main issue.
            logger.error(f"An error occurred in main execution: {e}", exc_info=True)
            retry_count += 1
            if retry_count <= max_retries:
                logger.info(f"Retrying ({retry_count}/{max_retries})...")
                asyncio.sleep(2) # Wait a moment before retrying
            else:
                logger.error("Maximum retry attempts reached. The application will now exit.")
                # Depending on the use case, you might want to handle this failure differently.