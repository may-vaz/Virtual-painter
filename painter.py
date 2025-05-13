import cv2
import numpy as np
import mediapipe as mp
import math

# Ask for camera permission
permission = input("Allow camera access? (y/n): ")
if permission.lower() != 'y':
    print("Camera access denied. Exiting.")
    exit()

# Initialize camera
cap = cv2.VideoCapture(0)

# Get canvas size from camera
ret, frame = cap.read()
if not ret:
    print("Failed to access camera.")
    cap.release()
    exit()

frame_height, frame_width = frame.shape[:2]
canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Variables for smoothing and color
prev_x, prev_y = None, None
smooth_x, smooth_y = None, None
smooth_factor = 0.4
min_movement = 4
current_color = (255, 0, 255)  # Default Pink

# Button configuration
button_size = 60
buttons = {
    'Red': (10, 10),
    'Green': (10, 80),
    'Blue': (10, 150),
    'Clear': (10, 220)
}

def check_button_click(x, y):
    global current_color, canvas
    for name, (bx, by) in buttons.items():
        if bx <= x <= bx + button_size and by <= y <= by + button_size:
            if name == 'Red':
                current_color = (0, 0, 255)
            elif name == 'Green':
                current_color = (0, 255, 0)
            elif name == 'Blue':
                current_color = (255, 0, 0)
            elif name == 'Clear':
                canvas[:] = 0

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for natural interaction
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Draw buttons
    for name, (bx, by) in buttons.items():
        color = (200, 200, 200) if name == 'Clear' else {'Red': (0, 0, 255), 'Green': (0, 255, 0), 'Blue': (255, 0, 0)}.get(name, (200, 200, 200))
        cv2.rectangle(frame, (bx, by), (bx + button_size, by + button_size), color, -1)
        cv2.putText(frame, name, (bx + 5, by + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Process hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            fingertip = hand_landmarks.landmark[8]
            new_x = int(fingertip.x * frame_width)
            new_y = int(fingertip.y * frame_height)

            check_button_click(new_x, new_y)

            if smooth_x is None and smooth_y is None:
                smooth_x, smooth_y = new_x, new_y
                prev_x, prev_y = new_x, new_y

            smooth_x = int(smooth_x * (1 - smooth_factor) + new_x * smooth_factor)
            smooth_y = int(smooth_y * (1 - smooth_factor) + new_y * smooth_factor)

            distance = math.hypot(smooth_x - prev_x, smooth_y - prev_y)
            if distance > min_movement:
                cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), current_color, 5)
                prev_x, prev_y = smooth_x, smooth_y

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        prev_x, prev_y = None, None
        smooth_x, smooth_y = None, None

    # Combine canvas and video frame
    combined = cv2.addWeighted(frame, 1, canvas, 0.5, 0)

    # Show output
    cv2.imshow("Virtual Painter", combined)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
