# --- START OF FILE danger_level_points.py ---

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Set

# Setup basic logging
logger = logging.getLogger(__name__)

# Define the structure for the obstacle category mapping
# Expects format like: {"Indoor Static": [56, 57,...], "Indoor Dynamic": [0, 15,...], ...}
ObstacleCategoryMapping = Dict[str, List[int]]

@dataclass
class DangerAssessment:
    """Contains results of danger assessment based on point system"""
    level: int                      # Final 1-4 (Low to Critical)
    description: str                # Text description of the level
    color: Tuple[int, int, int]     # BGR color for visualization
    requires_action: bool           # True if level is Warning or Critical
    total_points: int = 0           # Total points calculated
    points_breakdown: Dict[str, int] = field(default_factory=dict) # Points from each category
    distance: Optional[float] = None # Original distance input
    region: Optional[str] = None     # Original region input
    class_name: Optional[str] = None # Original class name input


class DangerLevelPointSystem:
    """
    Assesses danger levels based on a point system considering
    obstacle type, distance, and region.
    """

    # --- Point System Definitions ---
    POINTS_OBSTACLE_TYPE = {
        "Dynamic Outdoor": 4,
        "Dynamic Indoor": 3,
        "Static Indoor/Outdoor": 2, # Combined static category
        "General/Unknown": 1         # Default for non-obstacle classes or unknowns
    }

    # Distance thresholds and points (< threshold gets points)
    POINTS_DISTANCE = [
        (0.5, 4),  # < 0.5m = 4 points
        (1.0, 3),  # 0.5 <= d < 1.0m = 3 points
        (2.0, 2),  # 1.0 <= d < 2.0m = 2 points
    ] # Distance >= 2.0m gets 1 point implicitly later

    # Region keywords and points
    POINTS_REGION_CENTER_DOWN = 2
    POINTS_REGION_SIDE = 1
    POINTS_REGION_UP_OFFPATH = 0

    # --- Total Points to Final Level Mapping ---
    # Adjust these thresholds based on testing and desired sensitivity
    LEVEL_THRESHOLDS = [
        (8, 4), # 8+ points = Level 4 (Critical)
        (6, 3), # 6-7 points = Level 3 (Warning)
        (4, 2), # 4-5 points = Level 2 (Caution)
    ] # <= 3 points = Level 1 (Safe) implicitly

    # --- Static Definitions ---
    COLORS = {
        1: (0, 255, 0),    # Green - Safe
        2: (0, 255, 255),  # Yellow - Caution
        3: (0, 165, 255),  # Orange - Warning
        4: (0, 0, 255)     # Red - Critical
    }
    DESCRIPTIONS = {
        1: "Safe - Monitor environment",
        2: "Caution - Potential obstacle nearby",
        3: "Warning - Obstacle requires attention",
        4: "Critical - Immediate action may be needed"
    }

    CRITICAL_DISTANCE_OVERRIDE_M = 1.0  # FIXED: Increased from 0.5 to 1.0 meter

    def __init__(self,
                 obstacle_categories: ObstacleCategoryMapping,
                 detector_class_names: Dict[int, str]):
        """
        Initializes the DangerLevel system.

        Args:
            obstacle_categories: Dictionary mapping category names (e.g., "Indoor Static")
                                 to lists of COCO class indices. Must match detector's config.
            detector_class_names: Dictionary mapping COCO class indices to class names,
                                  obtained from the loaded YOLO model (model.names).
        """
        if not obstacle_categories or not detector_class_names:
             raise ValueError("Obstacle categories and detector class names must be provided.")

        self.obstacle_categories = obstacle_categories
        self.detector_class_names = detector_class_names
        self._build_class_to_type_map() # Create helper map for faster lookups

    def _build_class_to_type_map(self):
        """Builds a mapping from class_name (str) to obstacle type (str)."""
        self.class_to_type_map: Dict[str, str] = {}
        category_priority = [ # Order matters if class is in multiple lists
            "Dynamic Outdoor",
            "Dynamic Indoor",
            "Outdoor Static", # Treat both static as one for points
            "Indoor Static"
        ]

        # Create sets of indices for faster lookup
        category_indices: Dict[str, Set[int]] = {
            cat: set(indices) for cat, indices in self.obstacle_categories.items()
        }

        for class_idx, class_name in self.detector_class_names.items():
            assigned_type = "General/Unknown" # Default
            mapped_to_point_category = "General/Unknown"

            # Check categories in priority order
            if class_idx in category_indices.get("Dynamic Outdoor", set()):
                 assigned_type = "Dynamic Outdoor"
                 mapped_to_point_category = "Dynamic Outdoor"
            elif class_idx in category_indices.get("Dynamic Indoor", set()):
                 assigned_type = "Dynamic Indoor"
                 mapped_to_point_category = "Dynamic Indoor"
            # Combine static categories for point calculation
            elif class_idx in category_indices.get("Outdoor Static", set()) or \
                 class_idx in category_indices.get("Indoor Static", set()):
                 # Determine specific type for potential future use, but map to combined points category
                 if class_idx in category_indices.get("Outdoor Static", set()):
                      assigned_type = "Static Outdoor"
                 else:
                      assigned_type = "Static Indoor"
                 mapped_to_point_category = "Static Indoor/Outdoor" # Map both to the 2-point category

            # Store the mapping from class name to the *point category* string
            self.class_to_type_map[class_name.lower()] = mapped_to_point_category
            # print(f"Mapping: {class_name} -> {assigned_type} -> Points Category: {mapped_to_point_category}") # Debugging


    def _get_obstacle_type_points(self, class_name: Optional[str]) -> int:
        """Calculates points based on the obstacle type."""
        if class_name is None:
             return self.POINTS_OBSTACLE_TYPE["General/Unknown"]

        obstacle_point_type = self.class_to_type_map.get(class_name.lower(), "General/Unknown")
        return self.POINTS_OBSTACLE_TYPE.get(obstacle_point_type, 1) # Default to 1 if type somehow unknown

    def _get_distance_points(self, distance: Optional[float]) -> int:
        """Calculates points based on distance."""
        if distance is None or distance == float('inf'):
            return 0 # No distance info, no points added for it

        for threshold, points in self.POINTS_DISTANCE:
            if distance < threshold:
                return points

        # If distance >= last threshold (e.g., 2.0m)
        return 1 # Safe distance gets 1 point

    def _get_region_points(self, region_name: Optional[str]) -> int:
        """Calculates points based on the perspective region name."""
        if region_name is None:
            return 0 # Default to 0 if region is unknown

        region_lower = region_name.lower()

        if "center" in region_lower or "down" in region_lower:
             # Check it's not something like "Off-Path-Down" if that's possible
            if "off-path" not in region_lower:
                return self.POINTS_REGION_CENTER_DOWN
        elif "left" in region_lower or "right" in region_lower:
             # Exclude "Up" regions (like "Left-Up", "Right-Up") from side points if needed
            if "off-path" not in region_lower and "up" not in region_lower:
                return self.POINTS_REGION_SIDE

        # Includes "Up", "Off-Path", or unknown formats
        return self.POINTS_REGION_UP_OFFPATH

    def assess_danger(self,
                      distance: Optional[float],
                      region_name: Optional[str],
                      class_name: Optional[str]) -> DangerAssessment:
        """
        Enhanced danger assessment using the point system, with critical distance override.
        """
        type_points = self._get_obstacle_type_points(class_name)
        distance_points = self._get_distance_points(distance)
        region_points = self._get_region_points(region_name)
        total_points = type_points + distance_points + region_points

        # Determine level based on total points
        level_from_points = 1  # Default to Safe
        for threshold, level in self.LEVEL_THRESHOLDS:
            if total_points >= threshold:
                level_from_points = level
                break

        # FIXED: Enhanced critical distance override
        final_level = level_from_points
        critical_override_active = False
        
        if distance is not None and distance != float('inf'):
            # Very close objects should be at least warning level
            if distance < 0.8 and final_level < 3:
                final_level = 3
                critical_override_active = True
                logger.debug(f"Close distance override (<0.8m) triggered for {class_name or 'object'}. Level set to 3.")
            
            # Extremely close objects should be critical
            if distance < self.CRITICAL_DISTANCE_OVERRIDE_M and final_level < 4:
                final_level = 4
                critical_override_active = True
                logger.debug(f"Critical distance override (<{self.CRITICAL_DISTANCE_OVERRIDE_M}m) triggered for {class_name or 'object'}. Level set to 4.")

        # Get description and color based on the potentially overridden final_level
        description = self.DESCRIPTIONS.get(final_level, "Unknown Level")
        if critical_override_active:
            description += " (Distance Override)"
        color = self.COLORS.get(final_level, self.COLORS[1])
        requires_action = final_level >= 3

        points_breakdown = {
            "type": type_points,
            "distance": distance_points,
            "region": region_points
        }

        return DangerAssessment(
            level=final_level,
            description=description,
            color=color,
            requires_action=requires_action,
            total_points=total_points,
            points_breakdown=points_breakdown,
            distance=distance if distance != float('inf') else None,
            region=region_name,
            class_name=class_name
        )

    def get_color(self, level: int) -> Tuple[int, int, int]:
        """Get color for a given danger level."""
        return self.COLORS.get(level, self.COLORS[1])

# --- Standalone Test Block ---
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    print("--- DangerLevelPointSystem Standalone Test ---")

    # --- Define Mock Obstacle Categories (MUST MATCH DETECTOR CONFIG) ---
    # Example based on previous detector config
    MOCK_OBSTACLE_CATEGORIES: ObstacleCategoryMapping = {
        "Indoor Static": [56, 57, 58, 59, 60, 61, 72, 73],
        "Indoor Dynamic": [0, 14, 15, 16],
        "Outdoor Static": [9, 10, 11, 12, 13],
        "Outdoor Dynamic": [1, 2, 3, 5, 7, 15, 16, 24, 28, 36] # Note dog/cat are duplicated
    }

    # --- Define Mock Detector Class Names (MUST MATCH LOADED MODEL) ---
    # Example for COCO
    MOCK_DETECTOR_CLASS_NAMES: Dict[int, str] = {
         0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck',
         9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
         14: 'bird', 15: 'cat', 16: 'dog', 24: 'backpack', 28: 'suitcase', 36: 'skateboard',
         56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
         72: 'refrigerator', 73: 'sink', 80: 'unknown_coco' # Example of a class not in categories
         # Add more as needed from COCO 80
    }

    # --- Initialize ---
    try:
        danger_assessor = DangerLevelPointSystem(
            obstacle_categories=MOCK_OBSTACLE_CATEGORIES,
            detector_class_names=MOCK_DETECTOR_CLASS_NAMES
        )
        print("DangerLevelPointSystem initialized.")
    except ValueError as e:
        print(f"Initialization Error: {e}")
        exit()

    # --- Test Cases ---
    test_cases = [
        # class_name, distance, region_name                           # Expected Points (T+D+R=Total -> Level)
        ("person", 0.4, "Center-Close"),                             # Dynamic Indoor(3) + Dist(4) + Region(2) = 9 -> 4
        ("car", 0.8, "Left-Close"),                                  # Dynamic Outdoor(4) + Dist(3) + Region(1) = 8 -> 4
        ("chair", 1.5, "Center-Mid"),                                # Static(2) + Dist(2) + Region(2) = 6 -> 3
        ("bench", 5.0, "Right-Far"),                                 # Static(2) + Dist(1) + Region(1) = 4 -> 2
        ("cat", 0.6, "Down"),                                        # Dynamic Indoor(3) + Dist(3) + Region(2) = 8 -> 4 (Cat is dynamic)
        ("suitcase", 1.2, "Left-Mid"),                               # Dynamic Outdoor(4) + Dist(2) + Region(1) = 7 -> 3 (Suitcase dynamic outdoor)
        ("potted plant", 10.0, "Center-Far"),                          # Static(2) + Dist(1) + Region(2) = 5 -> 2
        ("person", 1.8, "Up"),                                       # Dynamic Indoor(3) + Dist(2) + Region(0) = 5 -> 2
        ("bicycle", None, "Center-Close"),                           # Dynamic Outdoor(4) + Dist(0) + Region(2) = 6 -> 3
        ("stop sign", 0.9, "Off-Path-Right"),                        # Static(2) + Dist(3) + Region(0) = 5 -> 2
        ("unknown_coco", 1.1, "Center-Mid"),                         # General(1) + Dist(2) + Region(2) = 5 -> 2
        (None, 0.4, "Center-Close"),                                 # General(1) + Dist(4) + Region(2) = 7 -> 3
        ("person", 3.5, "Right-Far"),                                # Dynamic Indoor(3) + Dist(1) + Region(1) = 5 -> 2
        ("person", 0.2, "Right-Down"),                               # Dynamic Indoor(3) + Dist(4) + Region(1) = 8 -> 4
    ]

    print("\n--- Running Test Cases ---")
    for i, (c_name, dist, region) in enumerate(test_cases):
        print(f"\n{i+1}. Input: Class='{c_name}', Dist={dist}, Region='{region}'")
        assessment = danger_assessor.assess_danger(dist, region, c_name)
        print(f"   Result: Level={assessment.level} ({assessment.description})")
        print(f"           Points={assessment.total_points} (Breakdown: {assessment.points_breakdown})")
        print(f"           Action Required: {assessment.requires_action}")
        # print(f"           Color: {assessment.color}") # Optional

    print("\n--- Test Finished ---")

# --- END OF FILE danger_level_points.py ---