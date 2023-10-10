import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic  # holistic model
mp_drawing = mp.solutions.drawing_utils  # drawing utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION
    image.flags.writeable = False
    results = model.process(image)  # make predictions
    image.flags.writeable = True  # set image to writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR CONVERSION BACKWARDS
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              # draw face connections
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2,
                                                     circle_radius=2))  # pose connections

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    # left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),

                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    # draw right hand connections


# initialized mediapipe model
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
    while cap.isOpened():
        # read feed
        ret, frame = cap.read()
        # make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        # draw landmarks
        draw_landmarks(image, results)
        # show screen
        cv2.imshow('SLT Program', image)
        # break the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

draw_landmarks(frame, results)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Extract keypoints
# results.pose_landmarks
# pose = []
# for res in results.pose_landmarks.landmark:
#    test = np.array([res.x, res.y, res.z, res.visibility])
#    pose.append(test)


pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                 results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
face = np.array([[res.x, res.y, res.z] for res in
                 results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
lh = np.array([[res.x, res.y, res.z] for res in
               results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
rh = np.array([[res.x, res.y, res.z] for res in
               results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, lh, rh])


# FOLDERS FOR DATA COLLECTION, NUMPY ARRRAYS
DATA_PATH = os.path.join('MP_Data')
# ACTIONS TO DETECT
actions = np.array(['tanga', 'bobo', 'pakyu'])
no_sequences = 30
# no of frames
sequence_length = 30

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                # read feed
                ret, frame = cap.read()
                # make detections
                image, results = mediapipe_detection(frame, holistic)
                # draw landmarks
                draw_landmarks(image, results)
                # collections
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 90), 4,
                                cv2.LINE_AA)
                    cv2.putText(image, 'COLLECTING FRAMES FOR {} SEQUENCE NUMBER {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4, cv2.LINE_AA)
                    cv2.imshow("OPENCV FEED", image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'COLLECTING FRAMES FOR {} SEQUENCE NUMBER {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4, cv2.LINE_AA)
                    cv2.imshow("OPENCV FEED", image)
                keypoints = extract_keypoints(results)
                np_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(np_path, keypoints)



        # break the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

