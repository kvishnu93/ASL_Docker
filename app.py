import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the CNN model and label encoder
print("Loading CNN model and label encoder...")
model = tf.keras.models.load_model("./ASL_CNN_Model.h5")
with open("./ASL_Label_Encoder.pickle", "rb") as f:
    label_encoder = pickle.load(f)
print("Model and label encoder loaded successfully!")

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.9)

# State variables for stability and sentence construction
predicted_text = ""
final_characters = ""
same_characters = ""
count = 0


def preprocess_landmarks(landmark_array):
    """Preprocess landmarks to match the input shape of the CNN model."""
    landmark_array = np.array(landmark_array).flatten()
    return landmark_array.reshape(1, 42, 1)


@app.route("/")
def home():
    """Render the HTML page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Predict the ASL alphabet from the video frame."""
    global predicted_text, final_characters, same_characters, count
    try:
        # Receive the frame from the client
        frame = request.files["frame"].read()
        np_frame = np.frombuffer(frame, np.uint8)
        img = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

        # Process the frame with Mediapipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                # Extract and preprocess landmarks
                data_aux = []
                x_, y_ = [], []
                for landmark in hand_landmark.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                min_x, min_y = min(x_), min(y_)
                for landmark in hand_landmark.landmark:
                    normalized_x = landmark.x - min_x
                    normalized_y = landmark.y - min_y
                    data_aux.extend([normalized_x, normalized_y])

                input_data = preprocess_landmarks(data_aux)

                # Predict the character using the CNN model
                prediction = model.predict(input_data)
                predicted_class = np.argmax(prediction, axis=1)
                predicted_character = label_encoder.inverse_transform(predicted_class)[0]

                # Stability logic
                if predicted_character not in ["4"]:  # Ignore "4" placeholder
                    predicted_text += predicted_character

                    if predicted_text[-1] != predicted_text[-2]:  # Reset if a new character is detected
                        count = 0
                        same_characters = ""
                    else:
                        same_characters += predicted_character
                        count += 1

                    if count == 30:  # Confirm character after 30 consecutive detections
                        if predicted_character.lower() == "del":  # Back Space
                            if final_characters:
                                final_characters = final_characters[:-1]
                        elif predicted_character.lower() == "space":  # Space
                            final_characters += " "
                        else:
                            final_characters += predicted_character

                        # Reset counters
                        count = 0
                        same_characters = ""

        return jsonify({"final_characters": final_characters})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
