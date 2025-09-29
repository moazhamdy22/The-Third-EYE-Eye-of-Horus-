# --- START OF FILE main_sos_gps.py ---

import os
import logging
import sys
# Import necessary modules from the sos package
from sos.location_GPS import LocationService
from sos.emergency_serviceSOS import EmergencyService

# It's good practice to use logging in critical modules
logger = logging.getLogger("SOS_System")

# Add current directory to Python path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    
class SOSSystem:
    def __init__(self):
        # Initialize services with environment variables or defaults
        twilio_sid = os.getenv('TWILIO_SID', 'Your_sid')
        twilio_token = os.getenv('TWILIO_TOKEN', 'Your_token')
        twilio_phone = os.getenv('TWILIO_PHONE', 'Your_phone')
        
        self.location_service = LocationService()
        self.emergency_service = EmergencyService(
            twilio_sid=twilio_sid,
            twilio_token=twilio_token,
            twilio_phone=twilio_phone,
            email=os.getenv('EMAIL', "your_mail"),
            email_password=os.getenv('EMAIL_PASSWORD', "your_mail_password"),
            recipient_email=os.getenv('RECIPIENT_EMAIL', "your_recipient_mail")
        )
        self.emergency_number = os.getenv('EMERGENCY_NUMBER', '+20##########')
        
        self._validate_configuration()

    def _validate_configuration(self):
        """Validate that required configuration is present."""
        if not self.emergency_service.twilio_phone:
            logger.error("CRITICAL: Twilio phone number not configured properly.")
            print("CRITICAL WARNING: Twilio phone number not configured properly")
            
    def handle_emergency(self, emergency_type, location_info):
        """Handle a single emergency action."""
        try:
            if emergency_type == "call":
                logger.info("Attempting to make emergency call...")
                return self.emergency_service.make_emergency_call(self.emergency_number, location_info)
            elif emergency_type == "sms":
                logger.info("Attempting to send emergency SMS...")
                return self.emergency_service.send_emergency_sms(self.emergency_number, location_info)
            elif emergency_type == "email":
                logger.info("Attempting to send emergency email...")
                return self.emergency_service.send_emergency_email(location_info)
            else:
                logger.warning(f"Unknown emergency type: {emergency_type}")
                return False
        except Exception as e:
            logger.error(f"Error handling emergency type {emergency_type}: {str(e)}", exc_info=True)
            return False

def trigger_full_sos_alert():
    """
    This is the main function called by the listener thread.
    It executes the complete SOS sequence.
    """
    print("\n--- EMERGENCY SOS TRIGGERED ---")
    logger.critical("EMERGENCY SOS TRIGGERED")

    try:
        sos_system = SOSSystem()
        
        # 1. Get Location
        print("Getting location...")
        logger.info("SOS: Getting location...")
        location_info = sos_system.location_service.get_location()
        
        if not location_info:
            print("ERROR: Could not get location. Alerts will be sent without location info.")
            logger.error("SOS: Could not get location.")
            # Create a dummy location_info to send generic alerts
            location_info = {
                'coordinates': ['N/A', 'N/A'],
                'address_en': "Location not available",
                'address_ar': "الموقع غير متوفر"
            }
        else:
            print(f"Location found: {location_info['address_en']}")
            logger.info(f"SOS: Location found: {location_info['address_en']}")

        # 2. Send all alerts sequentially
        print("Sending all emergency alerts...")
        
        sms_sent = sos_system.handle_emergency("sms", location_info)
        email_sent = sos_system.handle_emergency("email", location_info)
        call_made = sos_system.handle_emergency("call", location_info)

        print("\n--- SOS Sequence Summary ---")
        print(f"SMS Sent: {'Success' if sms_sent else 'Failed'}")
        print(f"Email Sent: {'Success' if email_sent else 'Failed'}")
        print(f"Call Initiated: {'Success' if call_made else 'Failed'}")
        logger.info(f"SOS Summary: SMS={'Success' if sms_sent else 'Failed'}, Email={'Success' if email_sent else 'Failed'}, Call={'Success' if call_made else 'Failed'}")
        print("--------------------------\n")

    except Exception as e:
        print(f"CRITICAL ERROR during SOS sequence: {str(e)}")
        logger.critical(f"CRITICAL ERROR during SOS sequence: {str(e)}", exc_info=True)


# --- NEW: STANDALONE TEST BLOCK ---
if __name__ == "__main__":
    # This block runs only when you execute `python main_sos_gps.py` directly.
    # It's used for testing the alert functionality.
    
    # Setup basic logging for the test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("--- Running main_sos_gps.py in standalone test mode ---")
    print("This will attempt to send REAL alerts (SMS, Email, Call).")
    
    # Safety confirmation
    confirm = input("Are you sure you want to trigger a full test alert? (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        print("\nConfirmation received. Triggering test alert...")
        trigger_full_sos_alert()
        print("--- Test complete. ---")
    else:
        print("\nTest cancelled by user.")