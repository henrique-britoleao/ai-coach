#####  Imports  #####
import mediapipe as mp
import numpy as np

from abc import ABC, abstractmethod
from enum import Enum, auto

from src.utils import calculate_angle, is_standing, get_coord

mp_pose = mp.solutions.pose

#####  Abstract classes and Enums  #####
class Stage(Enum):
    ECCENTRIC = auto()
    CONCENTRIC = auto()
    START = auto()

class Exercise(ABC):
    """Class to specify exercise angle parameters."""
    stage: Stage
    
    @abstractmethod
    def validate_concentric(self, landmarks) -> bool:
        """Decides whether the concentric part of a rep is valid based on the current body position."""
    @abstractmethod
    def validate_eccentric(self, landmarks) -> bool:
        """Decides whether the eccentric part of a rep is valid based on the current body position."""
    
    def set_stage(self, stage: Stage) -> None:
        self.stage = stage

#####  Exercise implementations  #####
class PushUp(Exercise):   
    def validate_eccentric(self, landmarks) -> bool:
        elbow_angle, hip_angle = self._extract_angles(landmarks)
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        if (elbow_angle > 160 and 
            hip_angle < 185 and 
            hip_angle > 170 and 
            hip[1] < wrist[1]):
            return True
        else:
            return False
    
    def validate_concentric(self, landmarks) -> bool:
        elbow_angle, hip_angle = self._extract_angles(landmarks)
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        if (elbow_angle < 90 and  
            hip_angle < 185 and 
            hip_angle > 170 and 
            hip[1] < wrist[1]):
        
            return True
        else: 
            return False
    
    def _extract_angles(self, landmarks):
        shoulder = get_coord(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        elbow = get_coord(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
        wrist = get_coord(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
        knee = get_coord(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
        hip = get_coord(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
        
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        hip_angle = calculate_angle(shoulder, hip, knee)
        
        return elbow_angle, hip_angle

class Squat(Exercise):
    def validate_eccentric(self, landmarks) -> bool:
        hip_angle, knee_angle = self._extract_angles(landmarks)

        if (hip_angle > 160 and 
            knee_angle > 160):
            return True
        else:
            return False
    
    def validate_concentric(self, landmarks) -> bool:
        hip_angle, knee_angle = self._extract_angles(landmarks)
            
        if (hip_angle < 100 and
            knee_angle < 100):
            return True
        else:
            return False
        
    def _extract_angles(self, landmarks): 
        shoulder = get_coord(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]) # TODO: use right as well 
        knee = get_coord(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
        hip = get_coord(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
        ankle = get_coord(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        
        hip_angle = calculate_angle(shoulder, hip, knee)
        knee_angle = calculate_angle(hip, knee, ankle)
        
        return hip_angle, knee_angle

class Dip(Exercise):
    def validate_eccentric(self, landmarks) -> bool:
        elbow_angle, standing_bool = self._extract_angles(landmarks)
        
        if (elbow_angle > 160 and 
            standing_bool):
            return True
        else:
            return False

    def validate_concentric(self, landmarks) -> bool:
        elbow_angle, standing_bool = self._extract_angles(landmarks)

        if (elbow_angle < 100 and 
            standing_bool):
            return True
        else:
            return False

    def _extract_angles(self, landmarks): 
        shoulder = get_coord(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]) # TODO: use right as well 
        wrist = get_coord(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
        elbow = get_coord(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
        hip = get_coord(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
        
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        standing_bool = is_standing(shoulder, hip)

        return elbow_angle, standing_bool
    
