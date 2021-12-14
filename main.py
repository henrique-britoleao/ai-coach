#####  Imports  #####

import cv2
import mediapipe as mp
import numpy as np

from src.utils import calculate_angle
from src.exercises import Exercise
from src.counter import Counter

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def run_counter(exercise: Exercise):
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    counter = Counter()
    stage = None

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                
                # Calculate angle
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                hip_angle = calculate_angle(shoulder, hip, knee)
                
                # Visualize angle
                #'''
                cv2.putText(image, str(elbow_angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                '''
                cv2.putText(image, str(hip), 
                            tuple(np.multiply(hip, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                '''

                # # validate eccentric phase
                # if stage == 'up':
                #     if exercise.validate_eccentric(landmarks):
                #         stage = 'down'
                # # validate concentric phase
                # if stage == 'up':
                #     if exercise.validate_concentric(landmarks):
                #         stage = 'up'
                #         counter.add_rep()
                        
                # Push-up
                if elbow_angle > 160 and hip_angle < 185 and hip_angle > 170 and hip[1] < wrist[1]:
                    stage = "down"
                if elbow_angle < 90 and stage =='down' and hip_angle < 185 and hip_angle > 170 and hip[1] < wrist[1]:
                    stage="up"
                    counter.add_rep()
                    print(counter)

                # Squat

                # Situp

            except:
                pass
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (65,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (60,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    print(calculate_angle(np.array([1, 2]), np.array([3, 4]), np.array([5, 6])))