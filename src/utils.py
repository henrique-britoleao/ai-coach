#####  Imports #####
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark
import numpy as np

#####  Utils  #####
def calculate_angle(a: np.array, b: np.array, c: np.array) -> float:
    radians = np.arccos(
        np.dot((a-b), (c-b)) / (np.linalg.norm(a - b) * np.linalg.norm(c - b))
    ) 
    angle = np.abs(radians * 180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle 


def get_coord(landmark: NormalizedLandmark) -> np.ndarray:
    x, y, z = landmark.x, landmark.y, landmark.z
    
    return np.array([x, y, z])
    

def is_standing(shoulder, hip, threshold=0.3) -> bool:
    if abs(hip[1] - shoulder[1]) > threshold:
        return True
    else:
        return False
