import dlib
import numpy as np
import face_recognition_models
from sklearn.svm import SVC
import streamlit as st

from src.database.db import get_all_students

from ultralytics import YOLO

@st.cache_resource
def load_dlib_models():
    detector = dlib.get_frontal_face_detector() 


    sp = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )

    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    return detector, sp, facerec

def get_face_embeddings(image_np):
    _, sp, facerec = load_dlib_models()
    faces = detect_faces_yolo(image_np)

    encodings= []

    for face in faces:
        shape = sp(image_np, face)
        face_descriptor = facerec.compute_face_descriptor(image_np, shape, 1) #128 embedding

        encodings.append(np.array(face_descriptor))
    return encodings

@st.cache_resource
def get_trained_model():
    X = []
    y = []


    student_db = get_all_students()

    if not student_db:
        return None
    
    for student in student_db:
        embedding = student.get('face_embedding')
        if embedding:
            X.append(np.array(embedding))
            y.append(student.get('student_id'))

    if len(X) ==0:
        return 0
    
    clf = SVC(kernel='linear', probability=True, class_weight='balanced')

    try:
        clf.fit(X, y)
    except ValueError:
        pass

    return {'clf': clf, 'X':X, "y":y}


def train_classifier():
    st.cache_resource.clear()
    model_data = get_trained_model()
    return bool(model_data)

def predict_attendance(class_image_np):
    encodings = get_face_embeddings(class_image_np)

    detected_student = {}


    model_data = get_trained_model()

    if not model_data:
        return detected_student, [], len(encodings)
    
    clf = model_data['clf']
    X_train = model_data['X']
    y_train = model_data['y']

    all_students = sorted(list(set(y_train)))

    for encoding in encodings:
        if len(all_students)>= 2:
            predicted_id= int(clf.predict([encoding])[0])
        else:
            predicted_id = int(all_students[0])

        student_embedding = X_train[y_train.index(predicted_id)]

        best_match_score = np.linalg.norm(student_embedding - encoding)

        resemblance_threshold = 0.6

        if best_match_score <= resemblance_threshold:
            detected_student[predicted_id] = True
    return detected_student, all_students, len(encodings)
@st.cache_resource
def load_yolo_face_model():
    return YOLO('yolov8n.pt')


def detect_faces_yolo(image_np, confidence=0.4):
    model = load_yolo_face_model()
    results = model(image_np, verbose=False, classes=[0])  # class 0 = person

    face_boxes = []
    h_img = image_np.shape[0]

    for r in results:
        for box in r.boxes:
            if box.conf[0] >= confidence:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                person_height = y2 - y1
                face_bottom = y1 + int(person_height * 0.25)

                face_boxes.append(dlib.rectangle(x1, y1, x2, min(face_bottom, h_img)))

    return face_boxes