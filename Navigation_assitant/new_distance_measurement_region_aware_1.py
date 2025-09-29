# --- START OF FILE new_distance_measurement_region_aware.py ---

import serial
import serial.tools.list_ports
import time
import logging
from typing import Optional, Dict, Union, Tuple, List # Added List
from dataclasses import dataclass

# --- Configuration & Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DistanceResult:
    """Contains distance measurement results from different methods"""
    distance: float         # Final calculated distance in meters. Can be float('inf') if unknown.
    confidence: float       # Confidence score (0-1)
    method: str             # 'ultrasonic', 'pixel', or 'none'
    pixel_distance: Optional[float] = None  # Pixel-based measurement (meters)
    ultrasonic_distance: Optional[float] = None # Ultrasonic sensor measurement (meters)

# --- Ultrasonic Sensor Handling ---
class UltrasonicSensor:
    """Interface for ultrasonic distance sensor (e.g., HC-SR04 via Arduino)."""
    # Default max range for typical HC-SR04 style sensors
    DEFAULT_MAX_RANGE_M: float = 4.0
    # Consider readings above this unreliable even if technically returned
    RELIABLE_MAX_RANGE_M: float = 3.5

    def __init__(self, port: Optional[str] = None, baudrate: int = 9600, timeout: float = 0.5):
        """
        Initialize ultrasonic sensor connection.

        Args:
            port: Serial port name (e.g., 'COM3', '/dev/ttyACM0'). If None, attempts auto-detect.
            baudrate: Serial baud rate.
            timeout: Serial read timeout in seconds.
        """
        self.port_name = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_port = None
        self.max_range = self.RELIABLE_MAX_RANGE_M
        self.last_valid_reading_m = None
        self.consecutive_failures = 0
        self.max_failures = 3
        # Connect initially but don't raise if it fails
        self.connect()

    def is_connected(self) -> bool:
        """Check if sensor is properly connected."""
        return self.serial_port is not None and self.serial_port.is_open

    def _auto_detect_arduino_port(self) -> Optional[str]:
        """Attempts to find a typical Arduino serial port."""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if ('arduino' in port.description.lower() or 
                'ch340' in port.description.lower() or 
                'usb-serial' in port.description.lower() or 
                port.vid == 0x2341):
                return port.device
        return None

    def connect(self) -> bool:
        """Attempt to connect to the ultrasonic sensor."""
        # Close any existing connection first
        self.close()
        
        try:
            # Auto-detect port if none specified
            if self.port_name is None:
                self.port_name = self._auto_detect_arduino_port()
                if self.port_name:
                    logger.info(f"Auto-detected Arduino on port: {self.port_name}")
                else:
                    logger.warning("Could not auto-detect Arduino port.")
                    return False

            logger.info(f"Attempting connection to ultrasonic on {self.port_name}")
            self.serial_port = serial.Serial(
                port=self.port_name,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            time.sleep(2)  # Allow connection to stabilize
            
            # Test the connection with a reading
            self.serial_port.reset_input_buffer()
            self.serial_port.write(b'd')
            response = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
            
            if response and response.startswith('Distance:'):
                logger.info(f"Successfully connected to ultrasonic on {self.port_name}")
                self.consecutive_failures = 0
                return True
            else:
                logger.warning(f"Connected but got invalid response: {response}")
                self.close()
                return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.close()
            return False

    def get_distance(self) -> Optional[float]:
        """Read distance measurement from sensor in meters."""
        if not self.is_connected():
            return None

        try:
            self.serial_port.reset_input_buffer()
            self.serial_port.write(b'd')
            line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
            
            if line and line.startswith('Distance:'):
                try:
                    distance_cm = float(line.split(':')[1].strip())
                    distance_m = distance_cm / 100.0
                    
                    if 0.02 < distance_m <= self.max_range:
                        self.last_valid_reading_m = distance_m
                        self.consecutive_failures = 0
                        return distance_m
                    else:
                        self.consecutive_failures += 1
                except ValueError:
                    self.consecutive_failures += 1
            else:
                self.consecutive_failures += 1

            # Check for too many failures
            if self.consecutive_failures >= self.max_failures:
                logger.warning("Too many consecutive failures, closing connection")
                self.close()
                return None

        except (serial.SerialException, OSError) as e:
            logger.error(f"Serial error: {e}")
            self.close()
            return None

        return None

    def close(self):
        """Close serial connection."""
        if self.serial_port:
            try:
                self.serial_port.close()
            except Exception as e:
                logger.error(f"Error closing port: {e}")
            finally:
                self.serial_port = None

# --- Distance Measurement Logic ---
class DistanceMeasurement:
    """
    Unified distance measurement using pixel-based and ultrasonic methods,
    aware of object region for prioritization.
    """
    ULTRASONIC_RECONNECT_INTERVAL_S: float = 5.0  # Wait 5 seconds between reconnection attempts

    def __init__(
        self,
        focal_length: float, # Needs calibration!
        class_names: List[str],
        object_heights_m: Dict[str, float], # Expected object heights in METERS
        use_ultrasonic: bool = True,
        ultrasonic_port: Optional[str] = None # Auto-detect if None
    ):
        """
        Initialize the distance measurement system.

        Args:
            focal_length: Camera focal length in pixels (CALIBRATE THIS).
            class_names: List of detectable object class names (must match detector output).
            object_heights_m: Dictionary mapping class names to their typical real-world heights in METERS.
            use_ultrasonic: Whether to attempt to use the ultrasonic sensor.
            ultrasonic_port: Serial port for the ultrasonic sensor (e.g., 'COM3', '/dev/ttyACM0'). Auto-detects if None.

        Raises:
            ValueError: If focal_length is invalid or object_heights_m is empty.
        """
        logger.info("Initializing DistanceMeasurement...")
        if focal_length <= 0:
            raise ValueError("Focal length must be positive.")
        if not object_heights_m:
            raise ValueError("object_heights_m dictionary cannot be empty.")

        self.focal_length = focal_length
        self.class_names = class_names
        self.object_heights_m = object_heights_m
        self.default_height_m = 1.0 # Fallback height if class unknown
        logger.info(f"Using focal length: {self.focal_length:.2f} pixels")
        logger.info(f"Known object heights (m): {len(self.object_heights_m)}")

        self.use_ultrasonic = use_ultrasonic
        self.ultrasonic = None
        self._last_ultrasonic_connect_attempt = 0.0

        # Initialize ultrasonic sensor if requested
        if use_ultrasonic:
            logger.info("Ultrasonic sensor enabled. Attempting initialization...")
            try:
                self.ultrasonic = UltrasonicSensor(port=ultrasonic_port)
                if self.ultrasonic.serial_port is None:
                     logger.warning("Ultrasonic sensor initialization failed (check port/connection). Will rely on pixel-based method.")
                     self.ultrasonic = None # Ensure it's None if init failed
                else:
                    logger.info("Ultrasonic sensor initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize ultrasonic sensor system: {e}", exc_info=True)
                self.ultrasonic = None
        else:
             logger.info("Ultrasonic sensor explicitly disabled.")

    def _get_ultrasonic_reading_with_reconnect(self) -> Optional[float]:
        """Attempt to get ultrasonic reading, reconnecting if necessary."""
        if not self.use_ultrasonic or not self.ultrasonic:
            return None

        now = time.monotonic()
        
        # If disconnected and cooldown has passed, try to reconnect
        if not self.ultrasonic.is_connected():
            if now - self._last_ultrasonic_connect_attempt >= self.ULTRASONIC_RECONNECT_INTERVAL_S:
                logger.info("Attempting ultrasonic reconnection...")
                self.ultrasonic.connect()
                self._last_ultrasonic_connect_attempt = now

        # Try to get reading if connected
        if self.ultrasonic.is_connected():
            return self.ultrasonic.get_distance()
        
        return None

    def _calculate_pixel_distance(
        self,
        pixel_height: float,
        class_name: str
    ) -> Optional[float]:
        """Calculate distance based on pixel height and known object height."""
        if pixel_height <= 0:
            return None

        real_height = self.object_heights_m.get(class_name)
        if real_height is None:
            logger.warning(f"No known height for class '{class_name}'. Using default height {self.default_height_m}m for pixel distance.")
            real_height = self.default_height_m

        if real_height <= 0:
             logger.warning(f"Known height for class '{class_name}' is invalid ({real_height}m). Cannot calculate pixel distance.")
             return None

        try:
            distance_m = (real_height * self.focal_length) / pixel_height
            return distance_m if distance_m > 0 else None # Return None if calculation yields non-positive distance
        except ZeroDivisionError:
            logger.warning("Cannot calculate pixel distance: ZeroDivisionError (pixel_height likely zero).")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calculating pixel distance: {e}")
            return None

    def _combine_measurements(
        self,
        pixel_distance: Optional[float],
        ultrasonic_distance: Optional[float],
        region_name: str # Region name from classifier (e.g., "Center-Close", "Left-Far")
    ) -> Tuple[float, float, str]:
        """
        Combines measurements based on region and availability.
        Prioritizes ultrasonic for 'Center' regions, otherwise uses pixel-based.

        Returns:
            Tuple: (final_distance_m, confidence, method_used)
                   final_distance_m can be float('inf') if no method is valid.
        """
        final_distance = float('inf') # Default to infinity (unknown)
        confidence = 0.0
        method = 'none'

        # --- Determine if the object is in a region where ultrasonic is primary ---
        # Assumes region names like "Center-Close", "Center-Mid", "Center-Far"
        is_center_region = "Center" in region_name

        # Add debug logging
        logger.debug(f"Combine Check: Region='{region_name}', IsCenter={is_center_region}, UltrasonicObj={self.ultrasonic is not None}, UltrasonicDist={ultrasonic_distance}")

        # --- Attempt Ultrasonic measurement (Primary for Center Region) ---
        if is_center_region and self.ultrasonic and ultrasonic_distance is not None:
            logger.debug(f"Prioritizing ultrasonic for '{region_name}' region. Distance: {ultrasonic_distance:.2f} m")
            final_distance = ultrasonic_distance
            confidence = 0.85 # Higher confidence for prioritized, direct measurement
            method = 'ultrasonic'
            return final_distance, confidence, method # Return early

        # --- Fallback to Pixel-based measurement ---
        # Used if:
        # 1. Not a center region
        # 2. Is center region, but ultrasonic failed/unavailable/out_of_range
        if pixel_distance is not None and pixel_distance > 0:
            logger.debug(f"Using pixel-based distance for '{region_name}' region. Distance: {pixel_distance:.2f} m")
            final_distance = pixel_distance
            # Base confidence for pixel method. Could be adjusted based on class height certainty later.
            confidence = 0.65
            method = 'pixel'
            # If this was a center region where ultrasonic *failed*, we might slightly reduce confidence
            if is_center_region and self.ultrasonic and ultrasonic_distance is None:
                confidence = 0.60 # Indicate fallback due to ultrasonic failure
                logger.debug("Pixel distance used as fallback for center region (ultrasonic failed/unavailable).")

        # If pixel distance also failed or wasn't calculated
        elif method == 'none':
            logger.warning(f"No valid distance measurement available for object in '{region_name}'.")
            # final_distance remains float('inf'), confidence 0.0, method 'none'

        return final_distance, confidence, method

    def measure_distance(
        self,
        pixel_height: float,        # Height of detected object bbox in pixels
        class_name: str,            # Name of the detected class
        region_name: str            # Region classification result (e.g., "Center-Close")
    ) -> DistanceResult:
        """
        Measure distance using available methods, considering the object's region.

        Args:
            pixel_height: Height of the object's bounding box in pixels.
            class_name: The detected class name (used for pixel-based height lookup).
            region_name: The spatial region classification of the object.

        Returns:
            DistanceResult object containing distance measurements and confidence.
        """
        # 1. Calculate potential pixel-based distance
        pixel_dist = self._calculate_pixel_distance(pixel_height, class_name)

        # 2. Get ultrasonic distance with potential reconnection
        ultra_dist = self._get_ultrasonic_reading_with_reconnect()

        # 3. Combine based on region and availability
        final_dist, conf, method = self._combine_measurements(
            pixel_distance=pixel_dist,
            ultrasonic_distance=ultra_dist,
            region_name=region_name
        )

        return DistanceResult(
            distance=final_dist,
            confidence=conf,
            method=method,
            pixel_distance=pixel_dist,
            ultrasonic_distance=ultra_dist
        )

    def close(self):
        """Clean up resources (close ultrasonic sensor port)."""
        logger.info("Closing DistanceMeasurement resources...")
        if self.ultrasonic:
            self.ultrasonic.close()
        logger.info("DistanceMeasurement closed.")

# --- Standalone Test ---
def get_default_object_heights() -> Dict[str, float]:
    """ Provides a default dictionary of object heights in meters. """
    return {
        'person': 1.75, 'car': 1.6, 'truck': 2.5, 'bus': 3.0,
        'bicycle': 1.1, 'motorcycle': 0.8, 'traffic light': 2.5,
        'fire hydrant': 0.8, 'stop sign': 2.0, 'parking meter': 1.2,
        'bench': 0.5, 'chair': 0.9, 'couch': 0.8, 'potted plant': 0.6,
        'dining table': 0.75, 'refrigerator': 1.8, 'door': 2.0, # Assuming standard door
        'cat': 0.3, 'dog': 0.7, 'backpack': 0.45, 'suitcase': 0.6,
        'skateboard': 0.15, 'default': 1.0 # Fallback
    }

# --- END OF FILE ---