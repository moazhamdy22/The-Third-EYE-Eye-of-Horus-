# --- START OF FILE sos_listener.py ---

import serial
import time
import logging
import sys
import os

# We only import this for the non-test-mode path
from sos.main_sos_GPS import trigger_full_sos_alert 


# Add current directory to Python path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    
    
# Configure logger for this module
logger = logging.getLogger("SOS_Listener")

# Configuration
ARDUINO_PORT = 'COM7'  # Change this to your Arduino's COM port
BAUD_RATE = 9600
RECONNECT_DELAY = 10  # Seconds to wait before trying to reconnect

def listen_for_sos_trigger(test_mode=False):
    """
    This function runs in a separate thread, continuously listening
    for the SOS signal from the Arduino.

    Args:
        test_mode (bool): If True, will not send real alerts. Instead, it
                          will print a confirmation to the console.
                          Defaults to False.
    """
    logger.info(f"SOS Listener thread started. Test Mode: {test_mode}")
    
    while True:
        try:
            logger.info(f"Attempting to connect to Arduino on {ARDUINO_PORT}...")
            with serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1) as arduino:
                logger.info("Successfully connected to Arduino.")
                print("\n[SOS System Ready] Listening for emergency button press.")
                time.sleep(2)
                arduino.flushInput()
                
                while True:
                    if arduino.in_waiting > 0:
                        try:
                            data = arduino.readline().decode('utf-8').strip()
                            
                            if data == "BUTTON_PRESSED":
                                logger.warning("SOS BUTTON PRESS DETECTED!")
                                print("\n!!! EMERGENCY BUTTON PRESS DETECTED !!!")
                                
                                # --- MODIFIED PART ---
                                if test_mode:
                                    print("--- TEST MODE: Alert Trigger Detected. No real alert will be sent. ---")
                                else:
                                    # This is the crucial call to our centralized SOS handler
                                    trigger_full_sos_alert()
                                # --- END MODIFIED PART ---
                                
                                time.sleep(5)
                                arduino.flushInput()
                                print("\n[SOS System Ready] Resuming listening for emergency button.")

                        except UnicodeDecodeError:
                            logger.warning("SOS Listener: Caught a UnicodeDecodeError, flushing input.")
                            arduino.flushInput()
                        except Exception as e:
                            logger.error(f"Error while reading from serial: {e}")
                            break

        except serial.SerialException as e:
            logger.error(f"Serial connection error: {e}. Retrying in {RECONNECT_DELAY} seconds.")
            print(f"Error connecting to Arduino on {ARDUINO_PORT}. Will retry. Check connection.")
            time.sleep(RECONNECT_DELAY)
        except Exception as e:
            logger.critical(f"An unexpected error occurred in the SOS listener: {e}", exc_info=True)
            time.sleep(RECONNECT_DELAY)

# --- NEW: STANDALONE TEST BLOCK ---
if __name__ == "__main__":
    # This block runs only when you execute `python sos_listener.py` directly.
    # It's used for testing the Arduino hardware connection.
    
    # Setup basic logging for the test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("--- Running SOS Listener in standalone mode with REAL ALERTS ---")
    print(f"Attempting to connect to Arduino on {ARDUINO_PORT}.")
    print("Press the button on the Arduino to trigger a REAL SOS alert.")
    print("WARNING: REAL ALERTS WILL BE SENT IN THIS MODE.")
    print("Press Ctrl+C to exit.")
    
    try:
        # Call the listener function with real alerts (test_mode=False)
        listen_for_sos_trigger(test_mode=False)
    except KeyboardInterrupt:
        print("\nSOS Listener stopped by user.")