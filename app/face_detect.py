import cv2
import os

# Path to the Haar cascade file
cascade_path = 'haarcascade_frontalface_default.xml'

# Check if the cascade file exists
if not os.path.isfile(cascade_path):
    raise FileNotFoundError(f"The file {cascade_path} does not exist.")

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cascade_path)

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Error handling if cascade classifier is not loaded correctly
if face_cascade.empty():
    raise Exception(f"Error loading cascade classifier xml file: {cascade_path}")

# Check if camera opened successfully
if not cap.isOpened():
    raise Exception("Could not open video device")

#Create a directory to save detected video device
save_faces = True
output_dir = "detected_faces"
if save_faces and not os.path.exist(output_dir):
    os.makedirs(output_dir)

frame_count = 0
face_detection_enabled =True

# Main loop to capture video and detect faces
while True:
    ret, img = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if face_detection_enabled:

        face = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(30, 30))
        face_count_text = f"Faces detected: {len(faces)}"
        cv2.putText(img, face_count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
        # save the face image to disk
        if save_faces:
            face_img = img[y:y+h, x:x+w]
            face_filename = f"{output_dir}/face_{frame_count}.png"
            cv2.imwrite(face_filename, face_img)

    
    frame_count += 1
    
    # Display the video with face detection
    cv2.imshow("Face detection", img)

    # Keyboard controls
    key = cv2.waitKey(10)
    
    # Press 'd' to toggle face detection on/off
    if key == ord('d'):
        face_detection_enabled = not face_detection_enabled
        print(f"Face detection {'enabled' if face_detection_enabled else 'disabled'}")
    # Press 'Esc' to exit the loop
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


