import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
# Initialize text-to-speech engine
engine = pyttsx3.init()
# Labels dictionary
labels_dict = {
    0: 'Bathroom', 1: 'family', 2: 'hello', 3: 'help', 4: 'house', 5: 'i love you',
    6: 'more', 7: 'no', 8: 'please', 9: 'repeat', 10: 'seven', 11: 'six',  12: 'sorry',
    13: 'thank you', 14: 'two', 15: 'what', 16: 'when', 17: 'who', 18: 'why', 19: 'yes',20: 'I',
    21: 'have',22: 'a',23: 'good',24: 'morning',25: 'because',26: 'how',27: 'How are you',28: 'feel',29: 'day'
}

while True:
    # Read frame from video capture
    ret, frame = cap.read()
    H, W, _ = frame.shape

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        data_aux = []

        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Pad data_aux to match the expected number of features (84)
        while len(data_aux) < 84:
            data_aux.append(0.0)

        # Predict using the model
        prediction = model.predict([np.asarray(data_aux)])

        # Get the predicted character
        predicted_character = labels_dict[int(prediction[0])]
        # Speak the predicted label
        engine.say(predicted_character)
        engine.runAndWait()

        # Draw rectangle around hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x1 = int(min([landmark.x * W for landmark in hand_landmarks.landmark])) - 10
                y1 = int(min([landmark.y * H for landmark in hand_landmarks.landmark])) - 10
                x2 = int(max([landmark.x * W for landmark in hand_landmarks.landmark])) + 10
                y2 = int(max([landmark.y * H for landmark in hand_landmarks.landmark])) + 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
