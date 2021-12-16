#####  Imports  ##### 
import mediapipe as mp
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from typing import List, Dict

from src.exercises import Exercise, PushUp, Dip, SitUp, Squat

#####  Set Gloabals  #####
MP_POSE = mp.solutions.pose

DETECTION_POINTS = [ # list of points used for the exercise classification task
    MP_POSE.PoseLandmark.NOSE,
    MP_POSE.PoseLandmark.LEFT_SHOULDER,
    MP_POSE.PoseLandmark.RIGHT_SHOULDER,
    MP_POSE.PoseLandmark.LEFT_ELBOW,
    MP_POSE.PoseLandmark.RIGHT_ELBOW,
    MP_POSE.PoseLandmark.LEFT_WRIST,
    MP_POSE.PoseLandmark.RIGHT_WRIST,
    MP_POSE.PoseLandmark.LEFT_PINKY,
    MP_POSE.PoseLandmark.RIGHT_PINKY,
    MP_POSE.PoseLandmark.LEFT_INDEX,
    MP_POSE.PoseLandmark.RIGHT_INDEX,
    MP_POSE.PoseLandmark.LEFT_THUMB,
    MP_POSE.PoseLandmark.RIGHT_THUMB,
    MP_POSE.PoseLandmark.LEFT_HIP,
    MP_POSE.PoseLandmark.RIGHT_HIP,
    MP_POSE.PoseLandmark.LEFT_KNEE,
    MP_POSE.PoseLandmark.RIGHT_KNEE,
    MP_POSE.PoseLandmark.LEFT_ANKLE,
    MP_POSE.PoseLandmark.RIGHT_ANKLE,
    MP_POSE.PoseLandmark.LEFT_HEEL,
    MP_POSE.PoseLandmark.RIGHT_HEEL,
    MP_POSE.PoseLandmark.LEFT_FOOT_INDEX,
    MP_POSE.PoseLandmark.RIGHT_FOOT_INDEX
]

EXERCISE_DICT = {
    'dips': Dip,
    'pushup': PushUp,
    'situp': SitUp,
    'squats': Squat
}

MODEL_PATH = './classification_model/Model/model2'
ENCODER_PATH = './classification_model/Model/encoder2.pkl'
POINTS_PATH = './classification_model/Model/extracted_points.pickle'
PREDICTION_PATH = './classification_model/Model/predicted_exercise.pickle'

#####  Prediction Functions  #####
def extract_exercise_detection_points(
    landmarks, 
    detection_points: List[MP_POSE.PoseLandmark]=DETECTION_POINTS
) -> pd.DataFrame:
    extracted_points = []

    for point in detection_points:
        extracted_points.append(landmarks[point.value].x)
        extracted_points.append(landmarks[point.value].y)

    extracted_array = np.array(extracted_points).reshape(1, -1)
    
    return pd.DataFrame(extracted_array)

def predict_exercise(extracted_points_df: pd.DataFrame, 
                     model, 
                     encoder, 
                     exercise_dict: Dict[str, Exercise]=EXERCISE_DICT
    ) -> Exercise:
    # normalize entries
    X = extracted_points_df.sub(extracted_points_df.mean(axis=1), axis = 0)
    # get predictions
    predictions = encoder.inverse_transform(np.argmax(model.predict(X),axis=1))
    # count number of time an exercise is predicted
    values, counts = np.unique(predictions, return_counts=True)

    exercise = values[np.argmax(counts)]

    return exercise_dict[exercise]

if __name__ == '__main__':
    model = load_model(MODEL_PATH) # exercise detection model

    with open(ENCODER_PATH,'rb') as f:
        encoder = pickle.load(f)

    with open(POINTS_PATH,'rb') as f:
        extracted_points_df = pickle.load(f)

    exercise = predict_exercise(extracted_points_df, model, encoder)() 
    
    with open(PREDICTION_PATH, 'wb') as f:
        pickle.dump(exercise, f)
