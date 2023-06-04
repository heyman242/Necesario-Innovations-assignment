import cv2
import dlib
import numpy as np
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
from django.shortcuts import render

# Load the pre-trained face and mouth detector models
face_detector = dlib.get_frontal_face_detector()
mouth_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Constants for mouth width calculation
MOUTH_WIDTH_THRESHOLD = 45  # 4 cm
FILTER_EFFECT = (0, 0, 255)  # Red color


# Function to calculate the vertical difference between upper and lower lips
def calculate_vertical_difference(shape):
    upper_lip = shape[51]
    lower_lip = shape[57]
    vertical_difference = abs(upper_lip[1] - lower_lip[1])
    return vertical_difference


# Function to apply the filter effect when the vertical difference is more than the threshold
def apply_filter(frame, shape):
    vertical_difference = calculate_vertical_difference(shape)
    if vertical_difference > MOUTH_WIDTH_THRESHOLD:
        cv2.putText(frame, 'Open Wide!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, FILTER_EFFECT, 2, cv2.LINE_AA)

        upper_lip = shape[51]
        lower_lip = shape[57]

        # Draw red dots on upper and lower lips
        cv2.circle(frame, tuple(upper_lip), 2, (0, 0, 255), -1)
        cv2.circle(frame, tuple(lower_lip), 2, (0, 0, 255), -1)

        # Apply filter effect to differentiate when the vertical difference is more than the threshold
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Apply desired filter effect here

    return frame


# Generator function to stream video frames
def stream_frames():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)
        for face in faces:
            shape = mouth_detector(gray, face)
            shape = np.array([[shape.part(i).x, shape.part(i).y] for i in range(shape.num_parts)], dtype=np.int32)
            frame = apply_filter(frame, shape)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


# render the video stream
@gzip.gzip_page
def live_stream(request):
    try:
        return StreamingHttpResponse(stream_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
    except GeneratorExit:
        pass


#the home page
def home(request):
    return render(request, 'home.html')
