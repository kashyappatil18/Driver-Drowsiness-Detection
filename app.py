from flask import Flask, render_template, Response, request, jsonify, redirect, url_for # type: ignore
import cv2 # type: ignore
import dlib # type: ignore
from imutils import face_utils # type: ignore
from pygame import mixer # type: ignore
import imutils # type: ignore

app = Flask(__name__)

# Initialize pygame mixer for alarm sound
mixer.init()
mixer.music.load("music.wav")  # Make sure this file exists

# Drowsiness detection setup
thresh = 0.25
frame_check = 40  # Adjusted for a longer detection delay
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure file exists

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)  # Open webcam
flag = 0
detection_active = False  # Track if detection is running

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/detection')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global detection_active
    detection_active = not detection_active
    return jsonify({'status': 'running' if detection_active else 'stopped'})

def generate_frames():
    global flag, detection_active
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = imutils.resize(frame, width=600)

        if detection_active:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            subjects = detect(gray, 0)

            for subject in subjects:
                shape = predict(gray, subject)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                ear = (cv2.norm(leftEye[1] - leftEye[5]) + cv2.norm(leftEye[2] - leftEye[4])) / (2.0 * cv2.norm(leftEye[0] - leftEye[3]))
                ear += (cv2.norm(rightEye[1] - rightEye[5]) + cv2.norm(rightEye[2] - rightEye[4])) / (2.0 * cv2.norm(rightEye[0] - rightEye[3]))
                ear /= 2.0

                # Draw eye contours
                cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

                # Drowsiness detection
                if ear < thresh:
                    flag += 1
                    if flag >= frame_check:
                        cv2.putText(frame, "ALERT!", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        mixer.music.play()
                else:
                    flag = 0

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True)