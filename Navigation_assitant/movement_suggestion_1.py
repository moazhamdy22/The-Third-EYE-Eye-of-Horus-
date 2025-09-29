# --- START OF FILE movement_suggestion.py ---

from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class MovementSuggestion:
    """Contains movement suggestion details"""
    primary_action: str                 # e.g., STOP, SLOW, TURN, CONTINUE, PROCEED
    secondary_action: Optional[str] = None # e.g., TURN, ADJUST, ASSESS
    direction: Optional[str] = None     # e.g., LEFT, RIGHT
    urgency: int = 1                    # 1 (Low/Green), 2 (Medium/Orange), 3 (High/Red)
    description: str = ""               # User-friendly explanation

class MovementAdvisor:
    """
    Provides movement suggestions based on danger assessment,
    perspective region classification, and distance.
    """

    # Define standard actions and directions
    ACTION_STOP = "STOP"
    ACTION_SLOW = "SLOW"
    ACTION_TURN = "TURN"
    ACTION_CONTINUE = "CONTINUE"
    ACTION_PROCEED = "PROCEED"
    ACTION_ADJUST = "ADJUST" # For slight path corrections

    DIRECTION_LEFT = "LEFT"
    DIRECTION_RIGHT = "RIGHT"

    # Urgency levels map roughly to danger colors
    URGENCY_LOW = 1    # Green
    URGENCY_MEDIUM = 2 # Orange
    URGENCY_HIGH = 3   # Red

    # Distance thresholds for refining critical actions
    CRITICAL_STOP_DISTANCE_M = 0.75 # Distance below which immediate STOP is mandatory

    def get_suggestion(self,
                      danger_level: int,              # 1-4 from DangerLevelPointSystem
                      is_path_obstacle: bool,         # From PerspectiveRegionClassifier
                      region_name: Optional[str],     # e.g., "Center-Close", "Off-Path-Left"
                      distance: Optional[float],      # Meters, can be None or float('inf')
                      class_name: Optional[str] = None # Optional: for potentially tailoring description
                      ) -> MovementSuggestion:
        """Generate movement suggestion based on the comprehensive situation."""

        # --- Rule 0: Handle Off-Path Objects ---
        # If an object is definitively off the path, generally advise continuing,
        # unless the danger level is unexpectedly very high (might indicate tuning issue).
        if not is_path_obstacle:
            if danger_level >= 3: # Unexpectedly high danger for off-path? Caution needed.
                 return MovementSuggestion(
                     primary_action=self.ACTION_CONTINUE,
                     urgency=self.URGENCY_MEDIUM,
                     description=f"Caution: Object detected {region_name or 'nearby'}, monitor environment."
                 )
            else: # Normal off-path case (Level 1 or 2)
                 return MovementSuggestion(
                     primary_action=self.ACTION_PROCEED,
                     urgency=self.URGENCY_LOW,
                     description=f"Object {region_name or 'detected'} is off path. Proceed."
                 )

        # --- Rules based on Danger Level (Assuming is_path_obstacle is True now) ---

        # --- Level 4: Critical ---
        if danger_level == 4:
            obstacle_desc = f"{class_name or 'Obstacle'}"
            # Immediate stop if extremely close
            if distance is not None and distance < self.CRITICAL_STOP_DISTANCE_M:
                evasion_direction = self._get_evasion_direction(region_name)
                return MovementSuggestion(
                    primary_action=self.ACTION_STOP,
                    secondary_action=self.ACTION_TURN,
                    direction=evasion_direction,
                    urgency=self.URGENCY_HIGH,
                    description=f"STOP! {obstacle_desc} directly ahead at {distance:.1f}m!"
                )
            # Otherwise, stop and prepare to turn
            else:
                evasion_direction = self._get_evasion_direction(region_name)
                dist_info = f"at {distance:.1f}m" if distance and distance != float('inf') else "close"
                return MovementSuggestion(
                    primary_action=self.ACTION_STOP,
                    secondary_action=self.ACTION_TURN,
                    direction=evasion_direction,
                    urgency=self.URGENCY_HIGH,
                    description=f"STOP! {obstacle_desc} {dist_info} ({region_name or 'ahead'}). Turn {evasion_direction}."
                )

        # --- Level 3: Warning ---
        if danger_level == 3:
            evasion_direction = self._get_evasion_direction(region_name)
            obstacle_desc = f"{class_name or 'Obstacle'}"
            dist_info = f"at {distance:.1f}m" if distance and distance != float('inf') else "ahead"

            # FIXED: For close distances at warning level, suggest turning immediately
            if distance is not None and distance < 1.0:
                return MovementSuggestion(
                    primary_action=self.ACTION_TURN,
                    direction=evasion_direction,
                    urgency=self.URGENCY_HIGH,
                    description=f"{obstacle_desc} close {dist_info} ({region_name or 'in path'}). Turn {evasion_direction} now."
                )
            else:
                return MovementSuggestion(
                    primary_action=self.ACTION_SLOW,
                    secondary_action=self.ACTION_TURN,
                    direction=evasion_direction,
                    urgency=self.URGENCY_MEDIUM,
                    description=f"Slow down. {obstacle_desc} {dist_info} ({region_name or 'in path'}). Turn {evasion_direction}."
                )

        # --- Level 2: Caution ---
        if danger_level == 2:
            # FIXED: Provide actual movement guidance for caution level
            evasion_direction = self._get_evasion_direction(region_name)
            location_hint = ""
            if region_name:
                 if "Center" in region_name or "Down" in region_name:
                     location_hint = "ahead in path"
                 elif "Left" in region_name:
                     location_hint = "to the left"
                 elif "Right" in region_name:
                     location_hint = "to the right"

            # For caution level, suggest adjustment based on distance
            if distance is not None and distance < 1.5:
                return MovementSuggestion(
                    primary_action=self.ACTION_ADJUST,
                    direction=evasion_direction,
                    urgency=self.URGENCY_LOW,
                    description=f"Adjust path {evasion_direction.lower()}. Obstacle {location_hint}."
                )
            else:
                return MovementSuggestion(
                    primary_action=self.ACTION_CONTINUE,
                    urgency=self.URGENCY_LOW,
                    description=f"Continue with caution. Potential obstacle {location_hint}."
                )

        # --- Level 1: Safe ---
        return MovementSuggestion(
            primary_action=self.ACTION_PROCEED,
            urgency=self.URGENCY_LOW,
            description="Path clear, proceed normally."
        )

    def _get_evasion_direction(self, region_name: Optional[str]) -> str:
        """Determine best evasion direction based on obstacle location description."""
        if region_name is None:
            return self.DIRECTION_RIGHT # Default if region unknown

        region_lower = region_name.lower()

        if "left" in region_lower:
            return self.DIRECTION_RIGHT # Obstacle on left, turn right
        elif "right" in region_lower:
            return self.DIRECTION_LEFT  # Obstacle on right, turn left
        else: # Assume Center, Down, or unknown format within path -> default evasion
             return self.DIRECTION_RIGHT

    def get_urgency_color(self, urgency: int) -> Tuple[int, int, int]:
        """Get BGR color for urgency level."""
        if urgency == self.URGENCY_HIGH:    # 3
            return (0, 0, 255)     # Red
        elif urgency == self.URGENCY_MEDIUM: # 2
            return (0, 165, 255)  # Orange
        else: # 1 or default
             return (0, 255, 0)    # Green


# --- Standalone Test Block ---
if __name__ == "__main__":
    print("--- MovementAdvisor Standalone Test ---")
    advisor = MovementAdvisor()

    test_cases = [
        # danger_level, is_path_obstacle, region_name, distance, class_name
        (4, True, "Center-Close", 0.4, "person"),      # Critical, very close -> STOP!
        (4, True, "Left-Close", 0.8, "car"),         # Critical, close left -> STOP! Turn RIGHT.
        (3, True, "Center-Mid", 1.5, "chair"),      # Warning, mid center -> SLOW, Turn RIGHT.
        (3, True, "Right-Mid", 1.8, "bicycle"),     # Warning, mid right -> SLOW, Turn LEFT.
        (2, True, "Center-Far", 4.0, "potted plant"), # Caution, far center -> CONTINUE
        (2, True, "Left-Far", 5.0, "person"),       # Caution, far left -> CONTINUE
        (1, True, "Center-Far", 10.0, "bench"),      # Safe, very far -> PROCEED
        (3, False, "Off-Path-Left", 2.0, "dog"),      # Warning level but Off Path -> CONTINUE (Rule 0)
        (4, True, "Down", 0.6, "suitcase"),          # Critical, very close down -> STOP!
        (3, True, "Right-Down", 1.1, "backpack"),    # Warning, down right -> SLOW, Turn LEFT.
        (2, False, "Off-Path-Right", 3.0, "tree"),    # Caution but Off Path -> PROCEED (Rule 0)
        (4, True, "Center-Mid", None, "unknown"),     # Critical, unknown distance -> STOP! Turn RIGHT.
        (3, True, None, 1.5, "person"),              # Warning, unknown region -> SLOW, Turn RIGHT (default).
    ]

    print("\n--- Running Test Cases ---")
    for i, (level, on_path, region, dist, name) in enumerate(test_cases):
        print(f"\n{i+1}. Input: Level={level}, OnPath={on_path}, Region='{region}', Dist={dist}, Class='{name}'")
        suggestion = advisor.get_suggestion(level, on_path, region, dist, name)
        print(f"   Result-> Primary: {suggestion.primary_action}")
        if suggestion.secondary_action: print(f"            Secondary: {suggestion.secondary_action}")
        if suggestion.direction: print(f"            Direction: {suggestion.direction}")
        print(f"            Urgency: {suggestion.urgency}")
        print(f"            Description: {suggestion.description}")

    print("\n--- Test Finished ---")

# --- END OF FILE movement_suggestion.py ---