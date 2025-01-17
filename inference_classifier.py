import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Initialize camera
cap = cv2.VideoCapture(0)  # Ensure this is the correct camera index

if not cap.isOpened():
    print("Error: Could not open video device")
    exit()

# Define label dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
               20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3',
               30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret or frame is None:
        print("Error: Failed to capture image")
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Ensure consistent feature extraction
            min_x, min_y = min(x_), min(y_)
            data_aux = [(x - min_x) for x in x_]
            data_aux += [(y - min_y) for y in y_]

            # Pad or truncate data_aux to ensure it has 42 features
            if len(data_aux) < 42:
                data_aux.extend([0] * (42 - len(data_aux)))
            elif len(data_aux) > 42:
                data_aux = data_aux[:42]

            prediction = model.predict([np.asarray(data_aux)])

            # Check the type of prediction output
            if isinstance(prediction[0], int):
                predicted_character = labels_dict[prediction[0]]
            else:
                predicted_character = prediction[0]

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
