import cv2
import os

haar_file = '/Users/thilinananayakkara/Downloads/haarcascade_frontalface_default.xml'
datasets = '/Users/thilinananayakkara/Desktop/datasets'
sub_data = 'ELON'  # just the subfolder name, not the full path

# Create path to save the dataset
path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

# Set the image size (width, height) for face dataset images
(width, height) = (130, 100)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(haar_file)

# Start capturing from the webcam
webcam = cv2.VideoCapture(0)

count = 1
while count <= 100:  # Capture 100 images
    print(f"Capturing image {count}...")
    
    # Capture frame-by-frame from the webcam
    ret, im = webcam.read()
    
    if not ret:
        print("Failed to capture image.")
        break

    # Convert the image to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    # Loop through each detected face and process it
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the face from the image and resize it
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        # Save the resized face to the dataset folder
        cv2.imwrite(f'{path}/{count}.png', face_resize)

        # Increment the image counter
        count += 1

    # Display the frame with the face detection rectangle
    cv2.imshow('OpenCV Face Detection', im)

    # Break if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    # Break after 100 images have been saved
    if count > 100:
        break

# Release the webcam and close the windows
webcam.release()
cv2.destroyAllWindows()
