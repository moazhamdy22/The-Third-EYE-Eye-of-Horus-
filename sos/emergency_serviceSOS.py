# --- START OF FILE emergency_serviceSOS.py ---

from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class EmergencyService:
    def __init__(self, twilio_sid, twilio_token, twilio_phone, 
                 email, email_password, recipient_email):
        self.twilio_client = Client(twilio_sid, twilio_token)
        self.twilio_phone = twilio_phone
        self.email = email
        self.email_password = email_password
        self.recipient_email = recipient_email

    def _create_maps_link(self, coords):
        """Create a Google Maps link from coordinates."""
        if coords and coords[0] != 'N/A':
            return f"https://www.google.com/maps?q={coords[0]},{coords[1]}"
        return "Not available"

    def make_emergency_call(self, to_number, location_info):
        """Make an emergency call with location information."""
        try:
            # --- REVISED CALL SCRIPT ---
            # Clear, concise, and provides immediate context. Repeats key info.
            address = location_info.get('address_en', 'Location not available')
            
            twiml = f"""
            <?xml version="1.0" encoding="UTF-8"?>
            <Response>
                <Say language="en-US" voice="alice">
                    This is an automated emergency alert from a Smart Assistance System for a visually impaired individual.
                    Immediate assistance may be required.
                </Say>
                <Pause length="1"/>
                <Say language="en-US" voice="alice">
                    The last known location is: {address}.
                </Say>
                <Pause length="1"/>
                <Say language="en-US" voice="alice">
                    I repeat, this is an automated emergency alert for a visually impaired person.
                    The location is: {address}. Please respond.
                </Say>
            </Response>
            """
            
            call = self.twilio_client.calls.create(
                twiml=twiml,
                to=to_number,
                from_=self.twilio_phone
            )
            print(f"Emergency call initiated - SID: {call.sid}")
            return True
        except Exception as e:
            print(f"Error making emergency call: {e}")
            return False

    def send_emergency_sms(self, to_number, location_info):
        """Send an emergency SMS with location information."""
        try:
            # --- REVISED SMS BODY ---
            maps_link = self._create_maps_link(location_info['coordinates'])
            
            sms_body = (
                "**URGENT: AUTOMATED SOS ALERT**\n"
                "An emergency button was pressed on an assistance device for a Blind/Visually Impaired (BVI) individual.\n\n"
                f"LOCATION: {location_info['address_en']}\n"
                f"COORDS: {location_info['coordinates'][0]}, {location_info['coordinates'][1]}\n"
                f"MAP: {maps_link}\n\n"
                "Please attempt to contact them or proceed to the location immediately."
            )

            message = self.twilio_client.messages.create(
                body=sms_body,
                to=to_number,
                from_=self.twilio_phone
            )
            print(f"Emergency SMS sent - SID: {message.sid}")
            return True
        except Exception as e:
            print(f"Error sending SMS: {e}")
            return False

    def send_emergency_email(self, location_info):
        """Send an emergency email with location information."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = self.recipient_email
            # --- REVISED EMAIL SUBJECT ---
            msg['Subject'] = "URGENT: Automated SOS Alert for Visually Impaired Individual"

            maps_link = self._create_maps_link(location_info['coordinates'])
            
            # --- REVISED EMAIL BODY ---
            body = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: sans-serif; line-height: 1.6; }}
                    .container {{ padding: 20px; border: 3px solid #D9534F; background-color: #FDF7F7; border-radius: 8px; }}
                    h1 {{ color: #A94442; }}
                    strong {{ color: #333; }}
                    .map-link {{ display: inline-block; padding: 10px 15px; background-color: #4285F4; color: white; text-decoration: none; border-radius: 5px; margin-top: 15px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>** URGENT: AUTOMATED SOS ALERT **</h1>
                    <p>This is an automated emergency notification. The panic button on a smart assistance device for a <strong>Blind or Visually Impaired (BVI) individual</strong> has been activated.</p>
                    <p>Immediate attention is required.</p>
                    <hr>
                    <h3>Location Details:</h3>
                    <ul>
                        <li><strong>English Address:</strong> {location_info['address_en']}</li>
                        <li><strong>Arabic Address:</strong> {location_info['address_ar']}</li>
                        <li><strong>Coordinates:</strong> {location_info['coordinates'][0]}, {location_info['coordinates'][1]}</li>
                    </ul>
                    <a href="{maps_link}" class="map-link">View on Google Maps</a>
                    <hr>
                    <p>Please attempt to contact the individual or proceed to the location immediately. Your prompt response is crucial.</p>
                </div>
            </body>
            </html>
            """
            
            # The email will be sent as HTML
            msg.attach(MIMEText(body, 'html'))

            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(self.email, self.email_password)
                server.send_message(msg)
                
            print("Emergency email sent successfully")
            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False

# --- STANDALONE TEST BLOCK (no changes needed here) ---
if __name__ == "__main__":
    # Test configuration
    emergency_service = EmergencyService(
        twilio_sid='Your_sid',
        twilio_token='Your_token',
        twilio_phone='Your_phone', # Replace with a valid number if testing
        email="Your_mail",
        email_password="Your_password",
        recipient_email="your_recipient_mail"
    )
    
    # Test with dummy location info
    test_location = {
        'coordinates': [30.0444, 31.2357],
        'address_en': "Tahrir Square, Cairo, Egypt",
        'address_ar': "ميدان التحرير، القاهرة، مصر"
    }

    print("--- Testing new alert formats ---")
    
    # Test each service
    print("\n--- Testing Email ---")
    emergency_service.send_emergency_email(test_location)

    print("\n--- Testing SMS ---")
    emergency_service.send_emergency_sms('+20##########', test_location) # Replace with your test number

    print("\n--- Testing Call ---")
    emergency_service.make_emergency_call('+20##########', test_location) # Replace with your test number