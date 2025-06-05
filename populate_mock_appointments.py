import json
import os
import uuid
import random
import sys # For logger setup if run standalone
from datetime import datetime, timedelta, time
from pathlib import Path
from loguru import logger

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
APPOINTMENTS_FILE = DATA_DIR / "appointments.json"

if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created data directory: {DATA_DIR}")

if not APPOINTMENTS_FILE.exists():
    with open(APPOINTMENTS_FILE, 'w') as f:
        json.dump([], f)
    logger.info(f"Created empty appointments file: {APPOINTMENTS_FILE}")

def load_appointments_from_file() -> list:
    try:
        with open(APPOINTMENTS_FILE, 'r') as f:
            content = f.read()
            return json.loads(content) if content else []
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading {APPOINTMENTS_FILE}: {e}. Returning empty list.")
        return []
    except Exception as e: # Catch any other potential error during file read
        logger.error(f"Unexpected error loading {APPOINTMENTS_FILE}: {e}. Returning empty list.")
        return []


def save_appointments_to_file(appointments: list):
    try:
        with open(APPOINTMENTS_FILE, 'w') as f:
            json.dump(appointments, f, indent=2)
        logger.info(f"Successfully saved {len(appointments)} total appointments to {APPOINTMENTS_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving appointments to {APPOINTMENTS_FILE}: {e}")
        return False

# Constants for mock data generation
APPOINTMENT_DURATION_MINUTES = 30
BUSINESS_START_HOUR = 9
BUSINESS_END_HOUR = 17

# Mock data pools
patient_names = ["Alice Smith", "Bob Johnson", "Carol Williams", "David Brown", "Eve Jones", "Frank Garcia", "Grace Miller", "Henry Davis", "Ivy Rodriguez", "Jack Wilson", "Katie Moore", "Leo Taylor", "Mia Anderson", "Noah Thomas", "Olivia Martinez", "Paul Robinson"]
appointment_reasons = ["Annual Check-up", "Flu Symptoms", "Follow-up Visit", "New Patient Consultation", "Routine Exam", "Knee Pain Assessment", "Back Pain Relief", "Persistent Headache", "Allergy Testing", "Vaccination Appointment", "Prescription Refill Query", "Physical Therapy Initial Eval"]

# More structured medical info pools
allergies_pool = [
    [], # None
    [{"name": "Peanuts"}],
    [{"name": "Pollen"}],
    [{"name": "Dust Mites"}],
    [{"name": "Shellfish"}],
    [{"name": "Penicillin"}],
    [{"name": "Cats"}, {"name": "Ragweed"}]
]
prescriptions_pool = [
    [], # None
    [{"medication": "Lisinopril", "dosage": "10mg QD"}],
    [{"medication": "Amoxicillin", "dosage": "250mg TID"}],
    [{"medication": "Metformin", "dosage": "500mg BID"}],
    [{"medication": "Atorvastatin", "dosage": "20mg Daily"}],
    [{"medication": "Albuterol Inhaler", "dosage": "PRN"}],
    [{"medication": "Ibuprofen", "dosage": "400mg QID PRN"}]
]
conditions_pool = [
    [], # None
    [{"name": "Hypertension"}],
    [{"name": "Asthma"}],
    [{"name": "Diabetes Type 2"}],
    [{"name": "Arthritis - Right Knee"}],
    [{"name": "Chronic Migraine"}],
    [{"name": "Seasonal Allergies"}]
]

# Date range
start_date = datetime(2025, 6, 5)
end_date = datetime(2025, 6, 9)

def generate_mock_appointments():
    new_mock_appointments = []
    current_date_iter = start_date

    logger.info(f"Generating mock appointments from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")

    while current_date_iter <= end_date:
        if current_date_iter.weekday() < 5:  # Monday to Friday
            current_slot_time = time(BUSINESS_START_HOUR, 0)
            day_end_time = time(BUSINESS_END_HOUR, 0)

            while current_slot_time < day_end_time:
                slot_datetime = datetime.combine(current_date_iter, current_slot_time)
                
                # Increased probability to get more booked slots for testing
                if random.random() < 0.6:  # 60% chance of booking a slot
                    patient_name = random.choice(patient_names)
                    appt_reason = random.choice(appointment_reasons)
                    
                    # Ensure we get a distinct list object for each appointment
                    appt_allergies = random.choice(allergies_pool).copy()
                    appt_prescriptions = random.choice(prescriptions_pool).copy()
                    appt_conditions = random.choice(conditions_pool).copy()

                    mock_appointment = {
                        "appointment_id": str(uuid.uuid4()),
                        "scheduled_datetime_iso": slot_datetime.isoformat(),
                        "duration_minutes": APPOINTMENT_DURATION_MINUTES,
                        "patient_name": patient_name,
                        "intent": appt_reason, # Using appt_reason for intent
                        "allergies": appt_allergies,
                        "prescriptions": appt_prescriptions,
                        "conditions": appt_conditions,
                        "twilio_call_sid": f"mock_sid_{str(uuid.uuid4())[:8]}",
                        "status": "confirmed"
                    }
                    new_mock_appointments.append(mock_appointment)
                
                current_slot_datetime_obj = datetime.combine(datetime.min, current_slot_time) + timedelta(minutes=APPOINTMENT_DURATION_MINUTES)
                current_slot_time = current_slot_datetime_obj.time()
        
        current_date_iter += timedelta(days=1)

    logger.info(f"Generated {len(new_mock_appointments)} new mock appointments.")

    logger.info("Loading existing appointments...")
    existing_appointments = load_appointments_from_file()
    logger.info(f"Found {len(existing_appointments)} existing appointments.")

    # Add new mock appointments to existing ones
    # This simple extend will add all new ones. If you run script multiple times, you'll get more.
    # For true "only add if slot is free", this script would need to use the conflict checker.
    # For now, just appending is fine for generating test data.
    final_appointments = existing_appointments + new_mock_appointments
    
    # A more robust way to avoid duplicates if script is run multiple times on the same file
    # is to check by time slot, but for mock data, ensuring unique IDs for new items is enough.
    # The previous duplicate check was flawed if existing_appointments itself had duplicates.
    # Let's ensure we don't re-add existing appointments and only add new unique IDs.
    
    output_appointments = []
    seen_appointment_keys = set() # To track by time+patient or a unique slot identifier

    for appt in existing_appointments: # Add all existing first
        key = (appt.get("scheduled_datetime_iso"), appt.get("patient_name")) # Example key
        if key not in seen_appointment_keys:
            output_appointments.append(appt)
            seen_appointment_keys.add(key)

    for appt in new_mock_appointments: # Then add new ones if their slot isn't taken by an existing one
        key = (appt.get("scheduled_datetime_iso"), appt.get("patient_name"))
        # This simple keying won't prevent two *different* new mock appts for the same slot.
        # For this script's purpose (generating varied data), it's acceptable.
        # A real system would check is_slot_conflicting before adding.
        if key not in seen_appointment_keys: # Basic check to avoid adding mock over existing by chance
             output_appointments.append(appt)
             # We don't add to seen_appointment_keys here for new mocks to allow multiple new mocks in same slot if desired.
             # If strict one-per-slot is needed, this logic needs is_slot_conflicting.

    logger.info(f"Saving a total of {len(output_appointments)} appointments.")
    save_appointments_to_file(output_appointments)

    logger.info("Mock appointment generation complete.")

if __name__ == "__main__":
    if not getattr(logger, 'handlers', None): # Check if logger has handlers
        logger.remove() # Remove default handler if it exists
        logger.add(sys.stderr, level="INFO")
    generate_mock_appointments()
