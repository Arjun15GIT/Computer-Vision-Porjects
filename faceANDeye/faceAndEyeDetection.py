import cv2
import numpy as np

# Load the Haarcascade XML files for face and eye detection
face_classifier = cv2.CascadeClassifier(
    r"C:\Users\lenovo\OneDrive\Desktop\TRAINING\AI ML\PYTHON\Python Projects\faceANDeye\haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(
    r"C:\Users\lenovo\OneDrive\Desktop\TRAINING\AI ML\PYTHON\Python Projects\faceANDeye\haarcascade_eye.xml")


# Function to detect faces and eyes
def face_detector(img, size=0.5):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return img

    for (x, y, w, h) in faces:
        x = x - 50
        w = w + 50
        y = y - 50
        h = h + 50
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_classifier.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    roi_color = cv2.flip(roi_color, 1)
    return roi_color


# Load an image from file and display with face and eye detection
image = cv2.imread(
    r"C:\Users\lenovo\OneDrive\Desktop\TRAINING\AI ML\PYTHON\Python Projects\faceANDeye\WhatsApp Image 2024-08-02 at 18.14.14_5982c0b0.jpg")
detected_image = face_detector(image)

# Resize the window
cv2.imshow('Face and Eye Detection', detected_image)
cv2.resizeWindow('Face and Eye Detection', 800, 600)  # Change size as needed
cv2.waitKey(0)
cv2.destroyAllWindows()

# Real-time face and eye detection using webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    detected_frame = face_detector(frame)
    cv2.imshow('Our Face Extractor', detected_frame)
    cv2.resizeWindow('Our Face Extractor', 800, 600)  # Change size as needed
    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()
