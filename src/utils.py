#####  Imports #####
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