import mediapipe.solutions.pose as mp_pose

from abc import ABC, abstractmethod
from enum import Enum, auto

from src.utils import calculate_angle

class Stage(Enum):
    ECCENTRIC = auto()
    CONCENTRIC = auto()

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
        
class PushUp(Exercise):   
    stage: Stage 
    
    def validate_concentric(self, landmarks) -> bool:
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
    
    def validate_eccentric(self, landmarks) -> bool:
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
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, # TODO: test with 3-D points
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        hip_angle = calculate_angle(shoulder, hip, knee)
        
        return elbow_angle, hip_angle