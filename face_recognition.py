import cv2
import numpy as np
import os

# Haar Cascade file path
haar_file = '/Users/thilinananayakkara/Downloads/haarcascade_frontalface_default.xml'

# Path to dataset
datasets = '/Users/thilinananayakkara/Desktop/datasets'

print('Training...')

# Initialize lists and other variables
(images, labels, names, id) = ([], [], {}, 0)

# Set the desired image size (uniform size for all images)
(width, height) = (130, 100)

# Walk through the dataset directory
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            
            # Attempt to read the image
            img = cv2.imread(path, 0)  # Read as grayscale
            if img is not None:
                # Resize the image to a fixed size (width, height)
                img = cv2.resize(img, (width, height))
                
                # Append the image and corresponding label
                images.append(img)
                labels.append(id)
            else:
                print(f"Warning: Failed to load image {path}")
        
        id += 1

# Convert images and labels to numpy arrays with the correct data type
images = np.array(images, dtype=np.uint8)
labels = np.array(labels, dtype=np.int32)

# Print debugging information
print(f"Number of images: {len(images)}")
print(f"Number of labels: {len(labels)}")
print(f"Unique labels: {set(labels)}")
print(f"Names: {names}")

# Initialize the LBPH face recognizer model
model = cv2.face.LBPHFaceRecognizer_create()

# Train the model on the dataset
model.train(images, labels)

print("Training completed.")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(haar_file)

# Start video capture (webcam)
webcam = cv2.VideoCapture(0)  # Change to 0 if needed

# Variables for counting and processing unknown faces
cnt = 0
confidence_threshold = 800  # You can adjust this value

while True:
    # Capture frame from webcam
    ret, im = webcam.read()
    if not ret:
        print("Failed to capture image.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through detected faces
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))  # Resize the face to fit model input

        # Predict the face using the trained model
        prediction = model.predict(face_resize)

        # Draw a rectangle around the face
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # If the confidence level is below the threshold (recognition)
        if prediction[1] < confidence_threshold:
            name = names[prediction[0]]
            confidence = prediction[1]
            cv2.putText(im, f'{name} - {confidence:.0f}', (x - 30, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            print(f"Recognized: {name} (Confidence: {confidence:.0f})")
            cnt = 0
        else:
            cnt += 1
            cv2.putText(im, 'Unknown', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            print("Unknown face detected")

        # If an unknown face persists for a certain number of frames, save the image
        if cnt > 100:
            print("Persistent unknown person detected")
            cv2.imwrite("unknown_person.jpg", im)
            cnt = 0

    # Display the video with recognized faces
    cv2.imshow('Face Recognition', im)

    # Break loop if 'Esc' key is pressed
    if cv2.waitKey(10) == 27:
        break

# Release the webcam and close windows
webcam.release()
cv2.destroyAllWindows()
