import requests
import asyncio
import winsdk.windows.devices.geolocation as wdg
from geopy.geocoders import Nominatim
from typing import Tuple, List, Dict, Optional
import folium
from datetime import timedelta, datetime
import os
import math
import json
import logging
import time
import cv2
import numpy as np
import urllib3
import pyperclip  # Add this import for clipboard functionality

# Suppress urllib3 warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- START OF REPLACEMENT CODE ---

# Add these new imports at the top of your file
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

class CVInputHandler:
    """Handles OpenCV-based input for GPS navigation system with full Arabic support."""
    
    def __init__(self):
        self.window_name = "GPS Navigation Input"
        self.window_width = 800
        self.window_height = 600
        self.bg_color = (30, 30, 30)
        self.text_color = (230, 230, 230)
        self.highlight_color = (100, 180, 255)
        
        # Initialize with basic OpenCV font as default
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_path = None
        
        # Try to load Arabic font if available
        try:
            # Check for common font files that might support Arabic
            possible_fonts = [
                "arabic_font.ttf",
                "arial.ttf", 
                "NotoSansArabic-Regular.ttf",
                "tahoma.ttf"
            ]
            
            font_loaded = False
            for font_file in possible_fonts:
                try:
                    self.font_path = font_file
                    self.font_large = ImageFont.truetype(self.font_path, 30)
                    self.font_medium = ImageFont.truetype(self.font_path, 24)
                    self.font_small = ImageFont.truetype(self.font_path, 18)
                    self.font_input = ImageFont.truetype(self.font_path, 22)
                    logger.info(f"Successfully loaded font: {self.font_path}")
                    font_loaded = True
                    break
                except IOError:
                    continue
            
            if not font_loaded:
                raise IOError("No suitable fonts found")
                
        except (IOError, ImportError):
            logger.warning("No Arabic font available, using basic OpenCV text rendering")
            # Reset to None to indicate fallback mode
            self.font_path = None

    # --- NEW HELPER FUNCTION TO DRAW TEXT WITH ARABIC SUPPORT ---
    def draw_text(self, img, text: str, pos: Tuple[int, int], font, color: Tuple[int, int, int], right_to_left: bool = False):
        """Draws text using Pillow for full Unicode support or OpenCV as fallback."""
        # If font failed to load, use the old OpenCV method
        if not self.font_path:
            # Basic fallback without Arabic support
            # Choose appropriate OpenCV font size based on the requested font
            if font == getattr(self, 'font_large', None):
                font_scale = 0.9
                thickness = 2
            elif font == getattr(self, 'font_medium', None):
                font_scale = 0.7
                thickness = 2
            elif font == getattr(self, 'font_small', None):
                font_scale = 0.5
                thickness = 1
            else:
                font_scale = 0.7
                thickness = 2
            
            cv2.putText(img, text, pos, self.font, font_scale, color, thickness)
            return img

        # Use Pillow for Arabic support
        try:
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            
            # Convert OpenCV image (NumPy array) to Pillow image
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            if right_to_left:
                # Get text size to position it from the right edge
                text_width, text_height = draw.textbbox((0, 0), bidi_text, font=font)[2:4]
                actual_pos = (self.window_width - pos[0] - text_width, pos[1])
                draw.text(actual_pos, bidi_text, font=font, fill=color)
            else:
                draw.text(pos, bidi_text, font=font, fill=color)
            
            # Convert back to OpenCV image
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            # If anything fails with Pillow/Arabic, fall back to OpenCV
            logger.warning(f"Arabic text rendering failed, using fallback: {e}")
            cv2.putText(img, text, pos, self.font, 0.7, color, 2)
            return img

    # --- THE REST OF THE CLASS IS MODIFIED TO USE THE NEW `draw_text` HELPER ---

    def create_input_window(self, title: str, prompt: str, current_input: str = "", options: List[str] = None, message: str = ""):
        img = np.full((self.window_height, self.window_width, 3), self.bg_color, dtype=np.uint8)
        y_offset = 50
        
        img = self.draw_text(img, title, (20, y_offset), getattr(self, 'font_large', None), self.text_color)
        y_offset += 50
        
        img = self.draw_text(img, prompt, (20, y_offset), getattr(self, 'font_medium', None), self.text_color)
        y_offset += 40
        
        cv2.rectangle(img, (20, y_offset), (self.window_width-20, y_offset+60), (200, 200, 200), -1)
        
        display_text = current_input
        if len(display_text) > 60:
            display_text = "..." + display_text[-57:]
        
        is_arabic = any('\u0600' <= c <= '\u06FF' for c in display_text)
        img = self.draw_text(img, display_text, (30, y_offset + 15), getattr(self, 'font_input', None), (0, 0, 0), right_to_left=is_arabic)
        y_offset += 80
        
        if message:
            color = (0, 0, 255) if "error" in message.lower() or "invalid" in message.lower() else (0, 255, 0)
            img = self.draw_text(img, message, (20, y_offset), getattr(self, 'font_small', None), color)
            y_offset += 30
        
        if options:
            img = self.draw_text(img, "Available options:", (20, y_offset), getattr(self, 'font_small', None), self.text_color)
            y_offset += 25
            for i, option in enumerate(options[:10]):
                img = self.draw_text(img, f"{i+1}. {option}", (40, y_offset), getattr(self, 'font_small', None), self.text_color)
                y_offset += 20
        
        instructions = [
            "Type text and press Enter to confirm",
            "Ctrl+V to paste, Backspace to delete",
            "Press 0 or ESC to go back"
        ]
        y_bottom = self.window_height - 80
        for instr in instructions:
            img = self.draw_text(img, instr, (20, y_bottom), getattr(self, 'font_small', None), (150, 150, 150))
            y_bottom += 25
        
        return img

    def get_number_choice(self, title: str, options: Dict[str, str], instructions: str = "") -> str:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
        
        while True:
            img = np.full((self.window_height, self.window_width, 3), self.bg_color, dtype=np.uint8)
            y_pos = 50
            
            img = self.draw_text(img, title, (20, y_pos), getattr(self, 'font_large', None), self.text_color)
            y_pos += 60
            
            for key, text in options.items():
                # Split multiline text and draw each line
                lines = text.split('\n')
                for i, line in enumerate(lines):
                    is_arabic = any('\u0600' <= c <= '\u06FF' for c in line)
                    # Indent subsequent lines
                    x_offset = 40 if i == 0 else 60
                    # For a line containing only arabic, draw it RTL
                    if is_arabic and not any('a' <= c.lower() <= 'z' for c in line):
                        img = self.draw_text(img, f"{key}. {line}" if i == 0 else line, (40, y_pos), getattr(self, 'font_medium', None), self.text_color, right_to_left=True)
                    else:
                        img = self.draw_text(img, f"{key}. {line}" if i == 0 else line, (x_offset, y_pos), getattr(self, 'font_medium', None), self.text_color)
                    y_pos += 35
                y_pos += 5 # Extra space between options
            
            y_pos += 20
            for line in instructions.split('\n'):
                img = self.draw_text(img, line, (20, y_pos), getattr(self, 'font_small', None), (150, 150, 150))
                y_pos += 25
            
            cv2.imshow(self.window_name, img)
            key_pressed = cv2.waitKey(0) & 0xFF
            char_key = chr(key_pressed)
            
            if char_key in options:
                cv2.destroyWindow(self.window_name)
                return char_key

    def confirm_action(self, message: str) -> bool:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
        
        while True:
            img = np.full((self.window_height, self.window_width, 3), self.bg_color, dtype=np.uint8)
            
            img = self.draw_text(img, "Confirmation", (20, 50), getattr(self, 'font_large', None), self.text_color)
            # Handle multiline confirmation messages
            y_pos = 120
            for line in message.split('\n'):
                 img = self.draw_text(img, line, (20, y_pos), getattr(self, 'font_medium', None), self.text_color)
                 y_pos += 35

            img = self.draw_text(img, "5. Confirm", (40, 240), getattr(self, 'font_medium', None), self.highlight_color)
            img = self.draw_text(img, "0. Back", (40, 280), getattr(self, 'font_medium', None), self.text_color)
            
            cv2.imshow(self.window_name, img)
            key_pressed = cv2.waitKey(0) & 0xFF
            char_key = chr(key_pressed)
            
            if char_key == '5':
                cv2.destroyWindow(self.window_name)
                return True
            elif char_key == '0':
                cv2.destroyWindow(self.window_name)
                return False

    # The get_text_input function does not need to change significantly as
# --- START OF REPLACEMENT CODE ---
# In class CVInputHandler:

    def get_text_input(self, title: str, prompt: str, options: List[str] = None) -> str:
        """
        Get text input from user, with direct typing support for both Arabic (Windows) and English.
        """
        current_input = ""
        message = "" # Start with a clean message
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
        
        while True:
            # create_input_window uses Pillow and will correctly render the text
            img = self.create_input_window(title, prompt, current_input, options, message)
            cv2.imshow(self.window_name, img)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == 13:  # Enter
                if current_input.strip():
                    cv2.destroyWindow(self.window_name)
                    return current_input.strip()
                else:
                    message = "Input is empty. Please enter text or press 0/ESC to go back."
                    
            elif key == 8:  # Backspace
                if current_input:
                    current_input = current_input[:-1]
                message = ""
                
            elif key == 22:  # Ctrl+V (kept as a robust backup)
                try:
                    pasted_text = pyperclip.paste()
                    current_input += pasted_text
                    message = "Pasted from clipboard."
                except Exception:
                    message = "Could not access clipboard."
                continue
                
            elif key == ord('0') and not current_input:
                cv2.destroyWindow(self.window_name)
                return "0"
                
            elif key == 27:  # ESC key
                cv2.destroyWindow(self.window_name)
                return "0"
                
            # --- THIS IS THE MAIN FIX FOR DIRECT ARABIC TYPING ---
            elif key >= 32:  # Any other printable character
                try:
                    # First, try to decode as cp1256 for Windows Arabic keyboard input
                    char = key.to_bytes(1, 'big').decode('cp1256')
                    current_input += char
                    message = "" # Clear any previous message
                except UnicodeDecodeError:
                    # If cp1256 fails, it's likely English or another standard character.
                    # Fall back to the default chr() function.
                    try:
                        current_input += chr(key)
                        message = "" # Clear any previous message
                    except:
                        # This is a final fallback for a key that is not decodable at all
                        message = "Unrecognized character."
            # --- END OF FIX ---

        cv2.destroyWindow(self.window_name)
        return current_input.strip() if current_input else "0"



class WalkingDirections:
    def __init__(self):
        logger.info("Initializing WalkingDirections...")
        start_time = time.time()
        
        # Multiple routing services for fallback
        self.routing_services = [
            {
                "name": "OSRM",
                "url": "http://router.project-osrm.org/route/v1/foot/{},{};{},{}?steps=true&overview=full&annotations=true",
                "timeout": 30
            },
            {
                "name": "OSRM_Demo", 
                "url": "https://routing.openstreetmap.de/routed-foot/route/v1/driving/{},{};{},{}?steps=true&overview=full",
                "timeout": 30
            }
        ]
        
        # Initialize geolocator with longer timeout to reduce warnings
        self.geolocator = Nominatim(user_agent="my_walking_app", timeout=10)
        self.saved_places_file = "saved_places.json"
        self.directions_folder = "saved_outputs/directions"
        self.saved_places = self.load_saved_places()
        self.cv_input = CVInputHandler()
        
        # Create directions folder if it doesn't exist
        if not os.path.exists(self.directions_folder):
            os.makedirs(self.directions_folder)

        # Add average walking speed in meters per minute (5 km/h = ~83.33 m/min)
        self.walking_speed = 83.33  # meters per minute

        init_time = time.time() - start_time
        logger.info(f"Initialization completed in {init_time:.2f} seconds")
        
    def load_saved_places(self) -> Dict:
        """Load saved places from file"""
        try:
            if not os.path.exists(self.saved_places_file):
                # If file doesn't exist, create an empty one
                with open(self.saved_places_file, 'w', encoding='utf-8') as f:
                    json.dump({}, f, ensure_ascii=False, indent=2)
                return {}

            with open(self.saved_places_file, 'r', encoding='utf-8') as f:
                saved_places = json.load(f)
                return saved_places
                
        except Exception as e:
            error_msg = f"Error loading saved places: {e}"
            logger.error(error_msg)
            print(error_msg)
            return {}

    def save_place(self, key: str, name: str, lat: float, lon: float, address_en: str, address_ar: str) -> bool:
        """Save a place to the saved places file"""
        try:
            # Load current places
            with open(self.saved_places_file, 'r', encoding='utf-8') as f:
                saved_places = json.load(f)
            
            # Add or update place
            saved_places[key] = {
                "name": name,
                "lat": lat,
                "lon": lon,
                "address_en": address_en,
                "address_ar": address_ar
            }
            
            # Save back to file
            with open(self.saved_places_file, 'w', encoding='utf-8') as f:
                json.dump(saved_places, f, ensure_ascii=False, indent=2)
            
            # Update current instance
            self.saved_places = saved_places
            return True
        except Exception as e:
            error_msg = f"Error saving place: {e}"
            logger.error(error_msg)
            print(error_msg)
            return False

    async def get_coords(self) -> Tuple[float, float]:
        """Get GPS coordinates using Windows location API"""
        start_time = time.time()
        logger.info("Getting GPS coordinates...")
        try:
            locator = wdg.Geolocator()
            pos = await locator.get_geoposition_async()
            elapsed = time.time() - start_time
            logger.info(f"GPS coordinates obtained in {elapsed:.2f} seconds")
            return pos.coordinate.latitude, pos.coordinate.longitude
        except Exception as e:
            logger.error(f"Error getting GPS coordinates: {e}")
            raise

    def get_current_location(self) -> Tuple[float, float, str, str]:
        """Get current location using device GPS"""
        try:
            print("Getting your current location. This may take a moment.")
            lat, lon = asyncio.run(self.get_coords())
            # Get both English and Arabic addresses
            location_en = self.geolocator.reverse(f"{lat}, {lon}", language='en')
            location_ar = self.geolocator.reverse(f"{lat}, {lon}", language='ar')
            print("Successfully retrieved your location.")
            return lat, lon, location_en.address, location_ar.address
        except PermissionError:
            error_msg = "Location access error. Please enable location services in your Windows settings."
            logger.error(error_msg)
            print("\n=== ❌ Location Access Error ===")
            print("Please follow these steps:")
            print("1. Open Windows Settings")
            print("2. Go to Privacy & Security > Location")
            print("3. Enable 'Location services'")
            print("4. Enable 'Let apps access your location'")
            print("5. Enable 'Let desktop apps access your location'")
            print("===============================")
            return None, None, None, None
        except Exception as e:
            error_msg = f"Could not get current location: {e}"
            logger.error(error_msg)
            print(f"\n=== ❌ Error ===")
            print(error_msg)
            print("===============")
            return None, None, None, None

    def extract_readable_address(self, address: str) -> str:
        """Extract a readable portion of the address including building number"""
        parts = [part.strip() for part in address.split(',')]
        # Include building number and first meaningful parts (street, area, city)
        building_num = next((part for part in parts if part.replace(',', '').strip().isdigit()), '')
        meaningful_parts = [part for part in parts[:3] if not part.isdigit()]
        if building_num:
            meaningful_parts.insert(0, building_num)
        return ', '.join(meaningful_parts)

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the Haversine distance between two points in kilometers"""
        # Convert latitude and longitude from degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        radius = 6371  # Radius of the Earth in kilometers
        
        # Calculate the distance
        distance = radius * c
        return distance

    def calculate_step_time(self, distance: float) -> int:
        """Calculate estimated time in minutes for a step based on distance"""
        # Add minimum time of 1 minute for very short distances
        return max(1, round(distance / self.walking_speed))

    def search_location(self, query: str, current_lat: float = None, current_lon: float = None) -> List[Dict]:
        """Search for locations matching the query and sort by distance if current location is provided"""
        start_time = time.time()
        logger.info(f"Searching for location: {query}")
        try:
            print(f"Searching for {query}. Please wait.")
            locations = self.geolocator.geocode(query, exactly_one=False, language='en')
            if not locations:
                print("No locations found matching your query.")
                return []
            
            print(f"Found {len(locations)} locations. Processing results.")
            results = []
            for loc in locations:
                # Get Arabic version of the address
                loc_ar = self.geolocator.reverse(f"{loc.latitude}, {loc.longitude}", language='ar')
                # Get English version to ensure English street names
                loc_en = self.geolocator.reverse(f"{loc.latitude}, {loc.longitude}", language='en')
                
                location_info = {
                    'address_en': loc_en.address,  # Use English address
                    'address_ar': loc_ar.address if loc_ar else loc.address,
                    'lat': loc.latitude,
                    'lon': loc.longitude
                }
                
                # Calculate distance if current location is provided
                if current_lat is not None and current_lon is not None:
                    distance = self.calculate_distance(current_lat, current_lon, loc.latitude, loc.longitude)
                    location_info['distance'] = distance
                
                results.append(location_info)
            
            # Sort by distance if available
            if current_lat is not None and current_lon is not None:
                results.sort(key=lambda x: x['distance'])
            
            elapsed = time.time() - start_time
            logger.info(f"Location search completed in {elapsed:.2f} seconds")
            logger.info(f"Found {len(results)} locations")
            print(f"Search complete. Found {len(results)} locations.")
            return results
        except Exception as e:
            error_msg = f"Error searching location: {e}"
            logger.error(error_msg)
            print(error_msg)
            return []

    def get_detailed_instruction(self, step: Dict) -> str:
        """Generate detailed instruction based on maneuver type and modifiers"""
        maneuver = step.get('maneuver', {})
        maneuver_type = maneuver.get('type', '')
        modifier = maneuver.get('modifier', '')
        street_name = step.get('name', '')
        
        # Get English street name if Arabic is detected - with timeout handling
        if any('\u0600' <= c <= '\u06FF' for c in street_name):
            try:
                # Try to get English name from reverse geocoding with timeout
                coords = maneuver.get('location', [0,0])
                if coords:
                    location = self.geolocator.reverse(f"{coords[1]}, {coords[0]}", language='en', timeout=5)
                    if location:
                        street_name = location.raw.get('address', {}).get('road', street_name)
            except Exception:
                # Ignore timeout and other errors, just use default name
                pass
                
        # If street name is empty or still in Arabic, use a descriptive name
        if not street_name or any('\u0600' <= c <= '\u06FF' for c in street_name):
            street_name = "the street"
        
        if maneuver_type == 'roundabout':
            exit_num = maneuver.get('exit', 1)
            return f"At roundabout, take exit {exit_num} onto {street_name}"
        elif maneuver_type == 'turn':
            if modifier in ['right', 'left']:
                return f"Turn {modifier} onto {street_name}"
            elif modifier == 'straight':
                return f"Continue straight on {street_name}"
            elif modifier == 'slight right':
                return f"Take a slight right onto {street_name}"
            elif modifier == 'slight left':
                return f"Take a slight left onto {street_name}"
        elif maneuver_type == 'depart':
            return f"Start on {street_name}"
        elif maneuver_type == 'arrive':
            return "Arrive at destination"
        elif maneuver_type == 'new name':
            return f"Continue onto {street_name}"
        
        # Default case
        if street_name:
            return f"Continue on {street_name}"
        return "Continue straight ahead"

    def get_walking_directions(self, start_lat: float, start_lon: float, 
                             end_lat: float, end_lon: float) -> Optional[Dict]:
        """Get walking directions between two points with fallback services"""
        start_time = time.time()
        logger.info("Calculating walking directions...")
        
        # Check if coordinates are valid
        if not all(isinstance(coord, (int, float)) for coord in [start_lat, start_lon, end_lat, end_lon]):
            logger.error("Invalid coordinates provided")
            print("❌ Invalid coordinates. Please check your locations.")
            return None
            
        # Check if coordinates are within reasonable bounds
        if not (-90 <= start_lat <= 90 and -180 <= start_lon <= 180 and 
                -90 <= end_lat <= 90 and -180 <= end_lon <= 180):
            logger.error("Coordinates out of valid range")
            print("❌ Coordinates are out of valid range.")
            return None
        
        print("Calculating walking directions. Please wait...")
        
        # Try each routing service
        for service in self.routing_services:
            try:
                logger.info(f"Trying {service['name']} routing service...")
                url = service["url"].format(start_lon, start_lat, end_lon, end_lat)
                logger.info(f"Request URL: {url}")
                
                response = requests.get(url, timeout=service["timeout"])
                logger.info(f"Response status: {response.status_code}")
                
                if response.status_code != 200:
                    logger.warning(f"{service['name']} returned status {response.status_code}")
                    continue
                
                # Check if response has content
                if not response.text.strip():
                    logger.warning(f"{service['name']} returned empty response")
                    continue
                
                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                    logger.warning(f"{service['name']} returned invalid JSON: {e}")
                    logger.warning(f"Response content: {response.text[:200]}...")
                    continue
                
                # Check if the response indicates success
                if data.get("code") != "Ok":
                    error_msg = data.get("message", "Unknown error")
                    logger.warning(f"{service['name']} routing failed: {error_msg}")
                    continue
                
                # Check if routes exist
                if not data.get("routes") or len(data["routes"]) == 0:
                    logger.warning(f"{service['name']} found no routes")
                    continue
                    
                route = data["routes"][0]
                
                # Validate route data
                if not route.get("legs") or len(route["legs"]) == 0:
                    logger.warning(f"{service['name']} route has no legs")
                    continue
                
                steps = []
                
                for leg in route["legs"]:
                    if not leg.get("steps"):
                        continue
                    for step in leg["steps"]:
                        instruction = self.get_detailed_instruction(step)
                        distance = step.get('distance', 0)
                        duration = self.calculate_step_time(distance)
                        
                        steps.append({
                            "instruction": instruction,
                            "distance": distance,
                            "duration": duration
                        })
                
                if not steps:
                    logger.warning(f"{service['name']} route has no valid steps")
                    continue
                
                # Calculate total duration based on our walking speed
                total_distance = route.get("distance", 0)
                total_duration = sum(step["duration"] for step in steps) * 60  # Convert to seconds
                
                elapsed = time.time() - start_time
                logger.info(f"Route calculated successfully using {service['name']} in {elapsed:.2f} seconds")
                logger.info(f"Total distance: {total_distance/1000:.2f}km")
                logger.info(f"Total duration: {total_duration/60:.0f} minutes")
                
                print(f"✅ Route found using {service['name']}!")
                print(f"📏 Total distance: {total_distance/1000:.1f} kilometers")
                print(f"⏱️  Estimated walking time: {total_duration/60:.0f} minutes")
                
                return {
                    "total_distance": total_distance,
                    "total_duration": total_duration,
                    "steps": steps,
                    "service_used": service['name']
                }
                
            except requests.exceptions.Timeout:
                logger.warning(f"{service['name']} request timed out")
                continue
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error with {service['name']}")
                continue
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error with {service['name']}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error with {service['name']}: {e}")
                continue
        
        # If all services failed, try a simple straight-line fallback
        logger.warning("All routing services failed, calculating straight-line directions")
        return self.create_straight_line_directions(start_lat, start_lon, end_lat, end_lon)
    
    def create_straight_line_directions(self, start_lat: float, start_lon: float, 
                                      end_lat: float, end_lon: float) -> Optional[Dict]:
        """Create simple straight-line directions as fallback"""
        try:
            # Calculate straight-line distance
            distance_km = self.calculate_distance(start_lat, start_lon, end_lat, end_lon)
            distance_m = distance_km * 1000
            
            # Calculate bearing (direction)
            bearing = self.calculate_bearing(start_lat, start_lon, end_lat, end_lon)
            direction = self.bearing_to_direction(bearing)
            
            # Estimated walking time (assuming 5 km/h)
            duration_minutes = distance_km / 5 * 60
            
            steps = [
                {
                    "instruction": f"Head {direction} towards your destination",
                    "distance": distance_m,
                    "duration": int(duration_minutes)
                },
                {
                    "instruction": "Arrive at destination",
                    "distance": 0,
                    "duration": 0
                }
            ]
            
            logger.info(f"Created straight-line fallback route: {distance_km:.2f}km, {duration_minutes:.0f} minutes")
            print("⚠️  Using straight-line directions (detailed routing unavailable)")
            print(f"📏 Straight-line distance: {distance_km:.1f} kilometers")
            print(f"⏱️  Estimated time: {duration_minutes:.0f} minutes")
            
            return {
                "total_distance": distance_m,
                "total_duration": duration_minutes * 60,
                "steps": steps,
                "service_used": "Straight-line fallback"
            }
            
        except Exception as e:
            logger.error(f"Failed to create fallback directions: {e}")
            print("❌ Could not calculate any directions between these locations.")
            return None
    
    def calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the bearing between two points"""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon_rad = math.radians(lon2 - lon1)
        
        y = math.sin(delta_lon_rad) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad)
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        # Normalize to 0-360 degrees
        return (bearing_deg + 360) % 360
    
    def bearing_to_direction(self, bearing: float) -> str:
        """Convert bearing to cardinal direction"""
        directions = [
            "north", "northeast", "east", "southeast",
            "south", "southwest", "west", "northwest"
        ]
        
        # Each direction covers 45 degrees
        index = round(bearing / 45) % 8
        return directions[index]

    def format_directions(self, directions: Dict, start_address: str, end_address: str) -> List[str]:
        """Format directions into a list of strings"""
        total_minutes = int(directions['total_duration'] / 60)
        service_used = directions.get('service_used', 'Unknown')
        
        formatted = ["**Walking Directions**"]
        formatted.append(f"Distance: {directions['total_distance']:.0f}m ({directions['total_distance']/1000:.1f}km)")
        formatted.append(f"Estimated time: {total_minutes} minutes")
        formatted.append(f"Routing service: {service_used}")
        formatted.append(f"\nFrom: {start_address}")
        formatted.append(f"To: {end_address}\n")
        
        for i, step in enumerate(directions['steps'], 1):
            if "Arrive at destination" in step['instruction']:
                formatted.append(f"**{i}.** {step['instruction']}")
            else:
                duration_text = f"{step['duration']} min" if step['duration'] else ""
                distance_text = f"{step['distance']:.0f}m" if step['distance'] else ""
                formatted.append(f"**{i}.** {step['instruction']} ({distance_text}, {duration_text})")
        
        if service_used == "Straight-line fallback":
            formatted.append("\n⚠️  Note: This is a straight-line route.")
            formatted.append("Please use local knowledge for actual navigation.")
        
        formatted.append(f"\nDirections courtesy of {service_used}")
        return formatted

    def save_directions(self, directions: Dict, start_address: str, dest_address: str, filename: str):
        """Save directions to a text file in the directions folder"""
        start_time = time.time()
        logger.info(f"Saving directions to {filename}")
        try:
            # Create full path including directions folder
            filepath = os.path.join(self.directions_folder, filename)
            
            formatted_directions = self.format_directions(directions, start_address, dest_address)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(formatted_directions))
            print(f"Directions saved to file {filename} in directions folder")
            
            elapsed = time.time() - start_time
            logger.info(f"Directions saved in {elapsed:.2f} seconds")
        except Exception as e:
            error_msg = f"Error saving directions: {e}"
            logger.error(error_msg)

# --- (Keep all your code from the top down to the end of the WalkingDirections class) ---
# The classes CVInputHandler and WalkingDirections are correct and do not need changes.
# The replacement starts from the UI helper functions.

# --- UI HELPER FUNCTIONS WITH AUDIO INTEGRATION ---

def select_from_paginated_list(cv_input: CVInputHandler, audio_handler, title: str, items: List[Tuple[str, str]], prompt: str) -> Optional[str]:
    """ A generic, paginated selection menu using OpenCV with audio feedback. """
    total_items = len(items)
    if total_items == 0:
        message = "No items to display."
        if audio_handler: audio_handler.speak(message)
        cv_input.get_number_choice(title, {'0': "Back"}, f"{message} Press 0 to go back.")
        return None

    if audio_handler:
        # Announce the purpose of the list and how to navigate
        audio_handler.speak(f"Please {prompt}. Use the number keys to select an item. Press 8 for next page, and 9 for previous page.")

    page = 0
    items_per_page = 8
    total_pages = (total_items + items_per_page - 1) // items_per_page

    while True:
        start_idx = page * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        current_page_items = items[start_idx:end_idx]

        options = {str(i + 1): text for i, (key, text) in enumerate(current_page_items)}
        
        # Announce the items on the current page
        if audio_handler:
            audio_handler.speak(f"Page {page + 1} of {total_pages}.")
            for i, (key, text) in enumerate(current_page_items):
                # Clean text for audio by removing address details if present
                clean_text = text.split('\n')[0].replace('_', ' ')
                audio_handler.speak(f"Press {i+1} for {clean_text}.")
        
        instructions = f"{prompt}\n"
        page_info = f"Page {page + 1}/{total_pages}"
        
        nav_instructions = []
        if page > 0:
            options['9'] = "Previous Page"
            nav_instructions.append("9=Prev")
        if page < total_pages - 1:
            options['8'] = "Next Page"
            nav_instructions.append("8=Next")
        options['0'] = "Back to Menu"
        nav_instructions.append("0=Back")
        
        instructions += f"({', '.join(nav_instructions)})"

        choice = cv_input.get_number_choice(f"{title} - {page_info}", options, instructions)

        if choice == '0':
            return None
        elif choice == '9' and page > 0:
            page -= 1
        elif choice == '8' and page < total_pages - 1:
            page += 1
        elif choice.isdigit() and 1 <= int(choice) <= len(current_page_items):
            selected_index = start_idx + int(choice) - 1
            selected_key, selected_text = items[selected_index]
            
            clean_selected_text = selected_text.split('\n')[0]
            if audio_handler:
                audio_handler.speak(f"You selected {clean_selected_text}. Press 5 to confirm, or 0 to go back.")
            
            if cv_input.confirm_action(f"Confirm selection:\n{selected_text[:50]}?"):
                if audio_handler: audio_handler.speak("Confirmed.")
                return selected_key

def get_destination_from_user(walker: 'WalkingDirections', audio_handler, current_lat: float, current_lon: float) -> Optional[Dict]:
    """ Handles getting a destination from the user with audio feedback. """
    dest_options = { '1': "Enter new location", '2': "Choose from saved places", '0': "Back to main menu" }
    
    if audio_handler: audio_handler.speak("How would you like to select your destination? Press 1 to enter a new location. Press 2 to choose from your saved places.")
    
    dest_choice = walker.cv_input.get_number_choice("Select Destination", dest_options, "Choose destination option")

    if dest_choice == '0': return None

    if dest_choice == '1': # Search for new location
        if audio_handler: audio_handler.speak("Please type the destination address or location name.")
        query = walker.cv_input.get_text_input("Enter Destination", "Enter destination address:")
        if not query or query == '0': return None
        
        if audio_handler: audio_handler.speak(f"Searching for {query}. Please wait.")
        
        locations = walker.search_location(query, current_lat, current_lon)
        if not locations:
            msg = "No locations found for your search."
            if audio_handler: audio_handler.speak(msg)
            walker.cv_input.get_number_choice("Search Result", {'0':"Back"}, msg)
            return None
            
        if audio_handler: audio_handler.speak(f"Found {len(locations)} possible matches. Please select the correct one.")
        loc_items = [(loc['address_en'], f"{loc['address_en'][:60]}... ({loc.get('distance', 0):.1f}km)") for loc in locations]
        selected_addr = select_from_paginated_list(walker.cv_input, audio_handler, "Select Correct Location", loc_items, "choose a location")
        
        if not selected_addr: return None
            
        selected_location = next((loc for loc in locations if loc['address_en'] == selected_addr), None)
        
        if audio_handler: audio_handler.speak("Would you like to save this location for future use? Press 5 to save, or 0 to skip.")
        if walker.cv_input.confirm_action("Save this location for future use?"):
            if audio_handler: audio_handler.speak("Please enter a short key for this place, like work, or gym.")
            key = walker.cv_input.get_text_input("Save Location", "Enter place key (e.g., 'work'):")
            if key and key != '0':
                if audio_handler: audio_handler.speak(f"Now enter the full name for the place with key {key}.")
                name = walker.cv_input.get_text_input("Save Location", "Enter place name:")
                if name and name != '0':
                    walker.save_place(key.lower(), name, selected_location['lat'], selected_location['lon'], selected_location['address_en'], selected_location['address_ar'])
                    if audio_handler: audio_handler.speak(f"Successfully saved {name}.")
        
        return selected_location

    elif dest_choice == '2': # Select from saved
        if not walker.saved_places:
            msg = "You have no saved places."
            if audio_handler: audio_handler.speak(msg)
            walker.cv_input.get_number_choice("Saved Places", {'0':"Back"}, msg)
            return None
            
        saved_items = [(key, f"{data['name']} ({key})") for key, data in sorted(walker.saved_places.items())]
        selected_key = select_from_paginated_list(walker.cv_input, audio_handler, "Select Saved Place", saved_items, "choose a saved place")
        
        if selected_key:
            return walker.saved_places[selected_key]
            
    return None

def handle_manage_places(walker: 'WalkingDirections', audio_handler):
    """ Controller for the 'Manage Saved Places' menu with audio feedback. """
    while True:
        menu_options = { '1': "View saved places", '2': "Add a new place", '3': "Delete a place", '0': "Back to main menu" }
        
        if audio_handler: audio_handler.speak("Manage saved places. Press 1 to view, 2 to add, 3 to delete, or 0 to go back.")
        choice = walker.cv_input.get_number_choice("Manage Saved Places", menu_options, "Choose an option")

        if choice == '0': break
        
        elif choice == '1': # View
            if not walker.saved_places:
                if audio_handler: audio_handler.speak("You have no saved places to view.")
                continue
            saved_items = [(key, f"{data['name']}\n   {data['address_en'][:50]}...") for key, data in sorted(walker.saved_places.items())]
            select_from_paginated_list(walker.cv_input, audio_handler, "Viewing Saved Places", saved_items, "view place details")
        
        elif choice == '2': # Add
            # This flow reuses parts of get_destination_from_user
            if audio_handler: audio_handler.speak("Let's add a new place. Please type the address to search for.")
            # ... The rest of the add logic from get_destination_from_user would go here ...
            print("Add place logic is complex, reusing destination flow is preferred.")
            if audio_handler: audio_handler.speak("Please use the navigation menus to search for and save new places.")
            time.sleep(2)
        
        elif choice == '3': # Delete
            if not walker.saved_places:
                if audio_handler: audio_handler.speak("There are no places to delete.")
                continue
            saved_items = [(key, f"{data['name']} ({key})") for key, data in sorted(walker.saved_places.items())]
            key_to_delete = select_from_paginated_list(walker.cv_input, audio_handler, "Delete a Place", saved_items, "select a place to delete")
            
            if key_to_delete:
                place_name = walker.saved_places[key_to_delete]['name']
                if audio_handler: audio_handler.speak(f"Are you sure you want to delete {place_name}? This cannot be undone. Press 5 to confirm.")
                if walker.cv_input.confirm_action(f"REALLY DELETE '{place_name}'?"):
                    del walker.saved_places[key_to_delete]
                    with open(walker.saved_places_file, 'w', encoding='utf-8') as f:
                        json.dump(walker.saved_places, f, ensure_ascii=False, indent=2)
                    if audio_handler: audio_handler.speak(f"{place_name} has been deleted.")

# --- Main Function with Audio Integration ---
def main(shared_audio_handler=None):
    start_time = time.time()
    audio_handler = shared_audio_handler
    
    if audio_handler:
        audio_handler.speak("Welcome to the OpenStreet Maps directions system.")
    else:
        print("WARNING: No audio handler provided. Running without audio feedback.")
    
    logger.info("Starting OpenStreet Maps Directions App")
    walker = WalkingDirections()
    
    if audio_handler: audio_handler.speak("Checking routing services.")
    
    # ... (Service check remains the same) ...
    test_passed = False
    for service in walker.routing_services:
        try:
            test_url = service["url"].format(0, 0, 0.001, 0.001)
            response = requests.get(test_url, timeout=5)
            if response.status_code == 200: test_passed = True
        except: pass
    
    if not test_passed and audio_handler:
        audio_handler.speak("Warning: Detailed routing is unavailable. The system will use straight-line fallback directions.")
    
    while True:
        options = {
            '1': "Get directions from my current location",
            '2': "Get directions between saved places", 
            '3': "Manage saved places",
            '4': "Enter start and end locations manually",
            '0': "Return to Navigation Assistant Menu"
        }
        
        if audio_handler: audio_handler.speak("OpenStreet Maps Directions Menu. Press 1 for directions from your current location. Press 2 to use saved places. Press 3 to manage places. Press 4 to enter locations manually. Or press 0 to return to the main assistant menu.")
        
        choice = walker.cv_input.get_number_choice("OpenStreet Maps Directions", options, "Choose an option")

        start_location, end_location = None, None

        try:
            if choice == "0":
                if audio_handler: audio_handler.speak("Are you sure you want to return to the main assistant menu? Press 5 to confirm.")
                if walker.cv_input.confirm_action("Return to main assistant menu?"):
                    if audio_handler: audio_handler.speak("Returning.")
                    break
                else:
                    if audio_handler: audio_handler.speak("Cancelled.")
                    continue

            elif choice == "1": # From current location
                if audio_handler: audio_handler.speak("Getting your current location. This may take a moment.")
                lat, lon, addr_en, addr_ar = walker.get_current_location()
                if not lat: 
                    if audio_handler: audio_handler.speak("Error: Could not get your current location. Please check location services and try again.")
                    walker.cv_input.get_number_choice("Location Error", {'0': "Continue"}, "Could not get current location.")
                    continue
                
                if audio_handler: audio_handler.speak(f"Your current location is near {walker.extract_readable_address(addr_en)}.")
                start_location = {'lat': lat, 'lon': lon, 'address_en': addr_en, 'address_ar': addr_ar, 'name': 'Current Location'}
                end_location = get_destination_from_user(walker, audio_handler, lat, lon)

            elif choice == "2": # From saved places
                if not walker.saved_places:
                    if audio_handler: audio_handler.speak("You have no saved places. Please add a place first.")
                    continue
                
                saved_items = [(key, f"{data['name']} ({key})") for key, data in sorted(walker.saved_places.items())]
                start_key = select_from_paginated_list(walker.cv_input, audio_handler, "Select Start Place", saved_items, "choose a starting place")
                if not start_key: continue
                start_location = walker.saved_places[start_key]

                end_key = select_from_paginated_list(walker.cv_input, audio_handler, "Select Destination", saved_items, "choose a destination")
                if not end_key: continue
                end_location = walker.saved_places[end_key]

            elif choice == "3": # Manage places
                handle_manage_places(walker, audio_handler)
                continue

            elif choice == "4": # Manual entry
                # This option is less ideal for VI users but kept for completeness
                if audio_handler: audio_handler.speak("Manual entry. Please type the starting address.")
                start_query = walker.cv_input.get_text_input("Start Search", "Enter starting address:")
                # ... (complex flow, simplified audio) ...
                if audio_handler: audio_handler.speak("Now please type the destination address.")
                end_query = walker.cv_input.get_text_input("Destination Search", "Enter destination address:")
                # ... This flow would need full audio just like get_destination_from_user ...
                if audio_handler: audio_handler.speak("Manual entry requires significant typing. Using other options is recommended.")
                continue

            # --- Common Routing Logic ---
            if start_location and end_location:
                if audio_handler: 
                    audio_handler.speak("Calculating walking directions. Please wait.")

                route = walker.get_walking_directions(
                    start_location['lat'], start_location['lon'],
                    end_location['lat'], end_location['lon']
                )

                if route:
                    # --- Step 1: Speak the summary ---
                    total_km = route['total_distance'] / 1000
                    total_min = int(route['total_duration'] / 60)
                    summary_msg = f"Route found. Total distance is {total_km:.1f} kilometers. Estimated walking time is about {total_min} minutes. The full directions have been saved to a text file for your reference."
                    
                    if audio_handler: 
                        audio_handler.speak(summary_msg)

                    # --- Step 2: Save the directions to a file ---
                    dest_name = end_location.get('name', 'destination').replace(' ', '_').replace('/', '_').replace('\\', '_')
                    filename = f"directions_to_{dest_name}.txt"
                    walker.save_directions(route, start_location['address_en'], end_location['address_en'], filename)
                    
                    # --- Step 3 (THE FIX): Format and print the detailed steps to the console ---
                    print(f"\n=== ✅ Directions Saved ===")
                    print(f"📝 File: {filename}")
                    print("========================")
                    
                    print("\n" + "="*25)
                    print("🗺️ Walking Directions 🗺️")
                    formatted_directions = walker.format_directions(route, start_location['address_en'], end_location['address_en'])
                    print("\n".join(formatted_directions))
                    print("="*25 + "\n")
                    
                    # --- Step 4 (NEW): Audio feedback for turn-by-turn directions ---
                    if audio_handler:
                        audio_handler.speak("Now reading turn by turn directions.")
                        for i, step in enumerate(route['steps'], 1):
                            instruction = step['instruction']
                            distance = step.get('distance', 0)
                            duration = step.get('duration', 0)
                            
                            if "Arrive at destination" in instruction:
                                audio_msg = f"Step {i}. {instruction}"
                            else:
                                if distance > 0:
                                    distance_text = f"{distance:.0f} meters" if distance >= 100 else f"{distance:.0f} meters"
                                    time_text = f"{duration} minute{'s' if duration != 1 else ''}" if duration > 0 else ""
                                    if time_text:
                                        audio_msg = f"Step {i}. {instruction}. Distance: {distance_text}. Time: {time_text}."
                                    else:
                                        audio_msg = f"Step {i}. {instruction}. Distance: {distance_text}."
                                else:
                                    audio_msg = f"Step {i}. {instruction}"
                            
                            audio_handler.speak(audio_msg)
                        
                        audio_handler.speak("End of directions. Have a safe journey!")
                    # --- END OF NEW AUDIO FEEDBACK ---

                else:
                    msg = "Could not calculate a route between these locations."
                    if audio_handler: audio_handler.speak(msg)
                    walker.cv_input.get_number_choice("Route Error", {'0': "Continue"}, msg)

        except KeyboardInterrupt:
            logger.info("Program interrupted by user")
            if audio_handler: audio_handler.speak("GPS system interrupted.")
            break
        except Exception as e:
            logger.error(f"An error occurred in the GPS main loop: {e}", exc_info=True)
            if audio_handler: audio_handler.speak(f"An error occurred. Please check the console.")
            continue

    if audio_handler:
        audio_handler.speak("Exiting the OpenStreet Maps directions system.")
    
    logger.info(f"OpenStreet Maps Directions session ended after {time.time() - start_time:.2f} seconds")

# --- (Cleanup function remains the same) ---
def cleanup():
    logger.info("Cleaning up resources...")
    """Clean up async resources safely"""
    try:
        if asyncio.get_event_loop().is_running():
            pending = asyncio.all_tasks()
            for task in pending:
                task.cancel()
    except RuntimeError:
        pass
    logger.info("Cleanup completed")

# --- Main Guard with Standalone Test ---
if __name__ == "__main__":
    test_audio_handler = None
    try:
        # Import audio handler for standalone testing
        from audio_feedback_vision_assitant import AudioFeedbackHandler
        print("Running OpenStreet Maps Directions module in standalone test mode with audio.")
        test_audio_handler = AudioFeedbackHandler()
        main(shared_audio_handler=test_audio_handler)
    except ImportError:
        print("Running OpenStreet Maps Directions module in standalone test mode WITHOUT audio.")
        main(shared_audio_handler=None)
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled error in standalone mode: {e}", exc_info=True)
    finally:
        if test_audio_handler:
            test_audio_handler.stop()
        cleanup()
        print("\n=== 👋 OpenStreet Maps Directions Test Finished ===")

