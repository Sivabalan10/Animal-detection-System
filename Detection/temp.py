from flask import Flask, redirect, render_template, request, url_for
import requests
import cv2
import threading
from PIL import Image
import numpy as np
import torch
import time

app = Flask(__name__)

ESP32_IP = "192.168.4.148"

# Global variable to store LED state


# Function to control LED based on the stored state
def control_led(led_state):
    
    if led_state:
        response = requests.get(f"http://{ESP32_IP}/on")
        if response.status_code == 200:
            return 'LED toggled successfully'
    else:
        response = requests.get(f"http://{ESP32_IP}/off")
        if response.status_code == 200:
            return 'LED toggled successfully'
        

# Function to start camera and perform object detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')  # custom trained model
model.conf = 0.50

cam = None
thread = None
camera_on = False


def start_camera():
    global cam, thread, camera_on
    camera_on = True
    led_status = 0
    old_timestamp = None
    videopath = "test3.mp4"
    cam = cv2.VideoCapture(videopath)
    
    while camera_on:
        ret, frame = cam.read()
        if not ret:
            break

        frame = frame[:, :, [2, 1, 0]]
        frame = Image.fromarray(frame)
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        result = model(frame, size=600)
        detection = result.xyxy[0]  # Get detection results

        # Iterate over each detection
        for det in detection:
            label = int(det[5])  # Get label index
            conf = det[4]  # Get confidence score
            if conf >= model.conf:
                # Detetction is True
                label_name = model.names[label]  # Get label name
                try:
                    # print(f'Detected: {label_name}')
                    control_led(True)
                    led_status = 1
                    old_timestamp = time.time()
                    # ALERT TO MOBILE

                except Exception as e:
                    print("line 66")
            else:
                print("Line 68")

        current_time = time.time()
        if led_status == 1:
            if current_time - old_timestamp >= 2:
                control_led(False)
                led_status = 0
                # Record details sent to fire base...
                print("Record sent successfully..")
        else:
            control_led(False)

        # Display result
        cv2.imshow('YOLO', np.squeeze(result.render()))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()



@app.route('/')
def index():
    return render_template('index.html')


# Flask route to turn on LED
@app.route('/turn-on')
def turn_on():
    control_led(True)
    return "IOT ON"

# Flask route to turn off LED
@app.route('/turn-off')
def turn_off():
    control_led(False)
    return "IOT OFF"

@app.route('/camera_on')
def camera_on_route():
    global thread
    if thread is None or not thread.is_alive():
        thread = threading.Thread(target=start_camera)
        thread.start()
    return 'Camera On'

@app.route('/camera_off')
def camera_off_route():
    global camera_on
    camera_on = False
    return 'Camera Off'

if __name__ == '__main__':
    app.run(host='0.0.0.0',port = 5000, debug=True)