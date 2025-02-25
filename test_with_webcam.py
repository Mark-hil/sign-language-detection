import pickle
import cv2
import mediapipe as mp
import numpy as np
import datetime

# Load model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: Model file not found.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set webcam resolution to the highest supported by your camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Adjust width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Adjust height

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)  # Allow up to 2 hands

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R',
    17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y', 24: 'Z',
    25: 'space', 26: 'del'
}

typed_text = ""
stable_character = ""
stability_counter = 0  # Counts how long the character remains the same
STABILITY_THRESHOLD = 25  # Number of frames to confirm character stability

# Function to save typed text to a file
def save_typed_text(text):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"typed_text_{timestamp}.txt"
    with open(filename, 'w') as f:
        f.write(text)
    print(f"Typed text saved to {filename}")

# Function to resize frame while maintaining aspect ratio
def resize_frame(frame, target_width=None, target_height=None):
    if target_width is None and target_height is None:
        return frame

    h, w = frame.shape[:2]

    if target_width is None:
        # Calculate width based on target height
        ratio = target_height / h
        dim = (int(w * ratio), target_height)
    elif target_height is None:
        # Calculate height based on target width
        ratio = target_width / w
        dim = (target_width, int(h * ratio))
    else:
        # Resize to exact dimensions (may distort the image)
        dim = (target_width, target_height)

    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

# Function to wrap text into multiple lines
def wrap_text(text, max_width, font, font_scale, thickness):
    lines = []
    current_line = ''

    for char in text:
        # Check if adding the character exceeds the max width
        (width, _), _ = cv2.getTextSize(current_line + char, font, font_scale, thickness)
        if width <= max_width:
            current_line += char
        else:
            lines.append(current_line)
            current_line = char
    if current_line:
        lines.append(current_line)
    return lines

# List to store wrapped lines of text
text_lines = []

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("⚠️ Camera frame not received. Check your webcam connection.")
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

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

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0] if isinstance(prediction[0], str) else labels_dict.get(int(prediction[0]), "?")
        else:
            predicted_character = "?"

        # Stability Check
        if predicted_character == stable_character:
            stability_counter += 1
        else:
            stable_character = predicted_character
            stability_counter = 0

        # If stable for required frames, add to text
        if stability_counter == STABILITY_THRESHOLD:
            if stable_character == "space":
                typed_text += " "
            elif stable_character == "del":
                typed_text = typed_text[:-1]
            else:
                typed_text += stable_character
            stability_counter = 0  # Reset stability

            # Update wrapped text lines
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            max_text_width = W - 100  # Maximum width for text before wrapping
            text_lines = wrap_text(typed_text, max_text_width, font, font_scale, thickness)

        # Draw Prediction Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, stable_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Progress Bar (Red = Unstable, Green = Stable)
        bar_width = int((stability_counter / STABILITY_THRESHOLD) * 200)

        # Dynamically change color (Red -> Yellow -> Green)
        if stability_counter < STABILITY_THRESHOLD:
            red = int((STABILITY_THRESHOLD - stability_counter) * 255 / STABILITY_THRESHOLD)
            green = int(stability_counter * 255 / STABILITY_THRESHOLD)
            blue = 0  # Keeping blue at 0 for simplicity
        else:
            red = 0
            green = 255
            blue = 0

        bar_color = (blue, green, red)  # OpenCV uses BGR instead of RGB

        # Draw the progress bar with dynamic color
        cv2.rectangle(frame, (50, 50), (50 + bar_width, 80), bar_color, -1)
        cv2.rectangle(frame, (50, 50), (250, 80), (255, 255, 255), 2)  # Outline

        # Display Predictions
        cv2.putText(frame, predicted_character, (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

    # Display Typed Text with Wrapping and Scrolling
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    line_height = 40  # Height of each line of text

    # Calculate the maximum number of lines that can fit in the frame
    max_lines = (H - 400) // line_height  # 400 is the starting Y position

    # If the number of lines exceeds the maximum, remove the oldest lines
    if len(text_lines) > max_lines:
        text_lines = text_lines[-max_lines:]

    # Display the text lines
    text_y = 400  # Starting Y position for the text
    for line in text_lines:
        cv2.putText(frame, line, (50, text_y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        text_y += line_height  # Move down for the next line

    # Resize the frame to fit the screen while maintaining aspect ratio
    frame = resize_frame(frame, target_width=1280)  # Adjust target width as needed

    cv2.imshow('Sign Language Typing', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        save_typed_text(typed_text)  # Save typed text before exiting
        break

cap.release()
cv2.destroyAllWindows()