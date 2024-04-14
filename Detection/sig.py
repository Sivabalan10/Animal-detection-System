from flask import Flask, redirect, render_template, request, url_for
import requests
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models.detection as detection
import threading
from PIL import Image
import numpy as np
import torch
import time
from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

app = Flask(__name__)

import requests

ESP32_IP = "192.168.137.124"


REQUEST_TIMEOUT = 5  # Timeout in seconds

def control_led(led_state):
    try:
        if led_state:
            print("working")
            response = requests.get(f"http://{ESP32_IP}/on", timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                print("working...")
                return 'LED toggled successfully'
        else:
            response = requests.get(f"http://{ESP32_IP}/off", timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                return 'LED toggled successfully'
    except requests.exceptions.Timeout:
        return 'Request timed out. Check ESP32 connection.'
    except requests.exceptions.RequestException as e:
        return f'Error: {e}'


cam = None
thread = None
camera_on = False
led_status = 0
labelnames = ['none']
old_timestamp = None

def start_camera():
    global cam, thread,camera_on
    

    if not firebase_admin._apps:
    # If not initialized, initialize the app with a unique name
        cred = credentials.Certificate("farm-6b70f-firebase-adminsdk-dnr8b-90c5e19203.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://farm-6b70f-default-rtdb.firebaseio.com/'
        })
    ref = db.reference('/')

    model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.hub.load('ultralytics/yolov5',
                               'yolov5s', pretrained=True)
    model.conf = 0.50

    def preprocess(frame):
        h, w, _ = frame.shape
        new_h = int(np.ceil(h / 32) * 32)
        new_w = int(np.ceil(w / 32) * 32)
        frame_resized = cv2.resize(frame, (new_w, new_h))

        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        input_tensor = transform(frame_resized).unsqueeze(0)
        return input_tensor

 
    def detect_animals(frame):
        global led_status,old_timestamp,labelnames
        frame = frame[:, :, [2, 1, 0]]
        frame = Image.fromarray(frame)
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        result = model(frame)
        detections = result.xyxy[0]
        if detections.tolist() != []:
            for det in detections:
                label = int(det[5])
                conf = det[4]
                if conf >= model.conf:
                    label_name = model.names[label]
                    labelnames[0] = label_name
                    try:
                        # Agricultural intruding animals
                        confirmed_animals = ["elephant","cow","bear","bird","tiger","lion","pig"]
                        label_name = model.names[label]
                        label_name = label_name.lower()
                        if label_name in confirmed_animals:
                            print("Detection found! - ",label_name)
                            control_led(True)
                            led_status = 1
                            old_timestamp = time.time()
                            # ALERT TO MOBILE
                            data = {
                                    'is_detected': "true"
                                }
                            ref.child('-Nv7k9_XUoJtnF6P5YQB').update(data)
                            datas = {
                                'Animal_detected': str(labelnames[0]),
                                'time_stamp': str(datetime.now().strftime("%Y-%m-%d %H:%M"))
                            }
                            ref.child('-NvCCQB4XAL572083hD7').update(datas)
                            print("Record sent successfully..")
                        else:
                            data = {
                                'is_detected': "false"
                            }
                            ref.child('-Nv7k9_XUoJtnF6P5YQB').update(data)
                            current_time = time.time()
                            if led_status == 1:
                                if current_time - old_timestamp >= 2:
                                    control_led(False)
                                    led_status = 0
                                    # ref.child('-Nv7k9_XUoJtnF6P5YQB').update(data)
                            else:
                                control_led(False)
                            print("Detection occur - Not confirmed animal :",label_name)
                            
                            

                    except Exception as e:
                        print("Exception occurred:", e)

                else:
                    print("Confidence lower than threshold")
                    
        else:
            data = {
                'is_detected': "false"
            }
            ref.child('-Nv7k9_XUoJtnF6P5YQB').update(data)
            current_time = time.time()
            if led_status == 1:
                if current_time - old_timestamp >= 2:
                    control_led(False)
                    led_status = 0
                    # ref.child('-Nv7k9_XUoJtnF6P5YQB').update(data)
                    
                
            else:
                control_led(False)  
                

        frame_np = np.array(frame)

        for det in detections:
            bbox = det[:4].cpu().numpy().astype(
                int)
            cv2.rectangle(frame_np, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]), (0, 255, 0), 2)

        return frame_np

    cap = cv2.VideoCapture(0)

    while camera_on:
        ret, frame = cap.read()

        processed_frame = detect_animals(frame)

        cv2.imshow('Animal Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return "Working"
    # return jsonify(response), 200



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
    global thread,camera_on
    camera_on = True
    if thread is None or not thread.is_alive():
        thread = threading.Thread(target=start_camera)
        thread.start()
    return 'Camera On'

@app.route('/camera_off')
def camera_off_route():
    global camera_on
    camera_on = False
    control_led(False)
    return 'Camera Off'

if __name__ == '__main__':
    app.run(host='0.0.0.0',port = 5005, debug=True)