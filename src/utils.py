#####  Imports #####
import numpy as np

#####  Utils  #####
# def calculate_angle(a: np.array, b: np.array, c: np.array) -> float:
#     radians = np.arccos(
#         np.dot((a-b), (c-b)) / (np.linalg.norm(a - b) * np.linalg.norm(c - b))
#     ) 
#     angle = np.abs(radians * 180.0/np.pi)
    
#     if angle > 180.0:
#         angle = 360 - angle
        
#     return angle 

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 
