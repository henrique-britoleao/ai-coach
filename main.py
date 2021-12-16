#####  Imports  #####
import tensorflow as tf
import mediapipe as mp
import pandas as pd
import pickle
import cv2

from src.exercises import Stage
from src.exercise_detection import (extract_exercise_detection_points, predict_exercise)
from src.counter import Counter

MODEL_PATH = './Model/model2'
ENCODER_PATH = './Model/encoder2.pkl'

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def run():
    cap = cv2.VideoCapture(0)

    # Initialize variables
    counter = Counter() # counter
    exercise = None 
    iter = 0 # iteration counter
    extracted_points_df = pd.DataFrame() # exercise detection df
    model = tf.keras.models.load_model(MODEL_PATH) # exercise detection model
    with open(ENCODER_PATH,'rb') as file:
        encoder = pickle.load(file)
    
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
                landmarks = results.pose_world_landmarks.landmark

                if not exercise: # detect exercise
                    extracted_points = extract_exercise_detection_points(landmarks)
                    extracted_points_df = extracted_points_df.append(extracted_points)
                
                    if iter%10 == 0 :
                        exercise = predict_exercise(extracted_points_df, model, encoder)() 
                        extracted_points_df = pd.DataFrame() # flush all stored points
                
                else: # count reps
                    # validate eccentric phase
                    if exercise.stage == Stage.START:
                        if exercise.validate_eccentric(landmarks):
                            exercise.set_stage(Stage.ECCENTRIC)
                    if exercise.stage == Stage.CONCENTRIC:
                        if exercise.validate_eccentric(landmarks):
                            exercise.set_stage(Stage.ECCENTRIC)
                            counter.add_rep()
                    # # validate concentric phase
                    if exercise.stage == Stage.ECCENTRIC:
                        if exercise.validate_concentric(landmarks):
                            exercise.set_stage(Stage.CONCENTRIC)
                        
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
            cv2.putText(image, exercise.stage.name, 
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
    run()