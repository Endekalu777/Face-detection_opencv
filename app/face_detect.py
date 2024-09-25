import cv2
import os
import time
import logging


logging.basicConfig(level=logging.INFO)

# Path to the Haar cascade file
cascade_path = 'haarcascade_frontalface_default.xml'

# Path to pre-trained model for face detection (Deep Learning)
prototxt_path = 'deploy.prototxt.txt'
weights_path = 'res10_300x300_ssd_iter_140000.caffemodel'

# Check if the pre-trained model files exist
if not os.path.isfile(prototxt_path) or not os.path.isfile(weights_path):
    raise FileNotFoundError("Model files not found. Ensure 'deploy.prototxt.txt' and 'res10_300x300_ssd_iter_140000.caffemodel' are in the same directory.")

# Load the face detection model using OpenCV's DNN module
net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

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

# Save the video with detected faces
save_video = True
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_with_detections.avi', fourcc, 20.0, (640, 480))

# Main loop to capture video and detect faces
logging.info("Starting video capture and face detection...")
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    h, w = frame.shape[:2]

     # Prepare the frame for face detection (deep learning-based)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    if face_detection_enabled:
        face_count = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Only consider detections with confidence > 0.5
            if confidence > 0.5:
                face_count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")

                # Draw bounding box and label on the image
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
                text = f"Face: {confidence * 100:.2f}%"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Save the detected face as an image file
                if save_faces:
                    face_img = frame[y:y1, x:x1]
                    face_filename = f"{output_dir}/face_{frame_count}_{i}.png"
                    cv2.imwrite(face_filename, face_img)
                    logging.info(f"Face saved: {face_filename}")

        # Add the face count text on the frame
        face_count_text = f"Faces detected: {face_count}"
        cv2.putText(frame, face_count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the video with face detection
    cv2.imshow("Face Detection (Press 'd' to toggle, 'Esc' to exit)", frame)

    # Save the video with detections (if enabled)
    if save_video:
        out.write(frame)

    frame_count += 1

    # Keyboard controls
    key = cv2.waitKey(10)

    # Press 'd' to toggle face detection on/off
    if key == ord('d'):
        face_detection_enabled = not face_detection_enabled
        logging.info(f"Face detection {'enabled' if face_detection_enabled else 'disabled'}")

    # Press 'Esc' to exit the loop
    if key == 27:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
logging.info("Face detection stopped and video capture ended.")