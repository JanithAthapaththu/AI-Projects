import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class FaceRecognitionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.configure(bg='#2C3E50')

        # Haar Cascade file path
        self.haar_file = '/Users/thilinananayakkara/Downloads/haarcascade_frontalface_default.xml'

        # Path to dataset
        self.datasets = '/Users/thilinananayakkara/Desktop/datasets'

        self.face_cascade = cv2.CascadeClassifier(self.haar_file)
        self.model = None

        # Create a canvas that can fit the webcam image
        self.canvas = tk.Canvas(window, width=720, height=700, bg='#34495E')
        self.canvas.pack(padx=10, pady=10)

        # Button to create dataset
        self.btn_create_dataset = ttk.Button(window, text="Create Dataset", command=self.create_dataset)
        self.btn_create_dataset.pack(pady=5)

        # Button to train model
        self.btn_train = ttk.Button(window, text="Train Model", command=self.train_model)
        self.btn_train.pack(pady=5)

        # Button to trigger face recognition
        self.btn_recognize = ttk.Button(window, text="Who is this Person?", command=self.recognize_face)
        self.btn_recognize.pack(pady=5)

        # Label to display recognition result
        self.lbl_result = ttk.Label(window, text="", font=('Helvetica', 14))
        self.lbl_result.pack(pady=10)

        # Capture video frames
        self.cap = cv2.VideoCapture(0)
        self.update()

        self.window.mainloop()

    def create_dataset(self):
        sub_data = tk.simpledialog.askstring("Input", "Enter name for the dataset:")
        if sub_data:
            path = os.path.join(self.datasets, sub_data)
            if not os.path.isdir(path):
                os.mkdir(path)

            (width, height) = (130, 100)
            count = 1

            while count <= 100:  # Capture 100 images
                ret, im = self.cap.read()
                if not ret:
                    print("Failed to capture image.")
                    break

                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 4)

                for (x, y, w, h) in faces:
                    cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    face = gray[y:y + h, x:x + w]
                    face_resize = cv2.resize(face, (width, height))
                    cv2.imwrite(f'{path}/{count}.png', face_resize)
                    count += 1

                cv2.putText(im, f"Capturing image {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.display(im)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            self.lbl_result.config(text=f"Dataset created for {sub_data}")

    def train_model(self):
        print('Training...')
        (images, labels, names, id) = ([], [], {}, 0)
        (width, height) = (130, 100)

        for (subdirs, dirs, files) in os.walk(self.datasets):
            for subdir in dirs:
                names[id] = subdir
                subjectpath = os.path.join(self.datasets, subdir)
                for filename in os.listdir(subjectpath):
                    path = os.path.join(subjectpath, filename)
                    img = cv2.imread(path, 0)
                    if img is not None:
                        img = cv2.resize(img, (width, height))
                        images.append(img)
                        labels.append(id)
                    else:
                        print(f"Warning: Failed to load image {path}")
                id += 1

        images = np.array(images, dtype=np.uint8)
        labels = np.array(labels, dtype=np.int32)

        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.model.train(images, labels)
        self.names = names
        print("Training completed.")
        self.lbl_result.config(text="Model trained successfully")

    def recognize_face(self):
        if self.model is None:
            self.lbl_result.config(text="Please train the model first")
            return

        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (130, 100))
                prediction = self.model.predict(face_resize)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                if prediction[1] < 800:
                    name = self.names[prediction[0]]
                    confidence = prediction[1]
                    result_text = f"Recognized: {name} (Confidence: {confidence:.0f})"
                else:
                    result_text = "Unknown Person"
                cv2.putText(frame, result_text, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                self.lbl_result.config(text=result_text)
                break
            else:
                self.lbl_result.config(text="No face detected")
            self.display(frame)

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            self.display(frame)
        self.window.after(10, self.update)

    def display(self, frame):
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

# Create a window and pass it to the Application object
FaceRecognitionApp(tk.Tk(), "Face Recognition System")
