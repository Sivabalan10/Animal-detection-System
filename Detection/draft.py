from flask import Flask, request, jsonify
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models.detection as detection
from PIL import Image
import numpy as np
from datetime import datetime
# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import db




def animal_detection():
    # cred = credentials.Certificate(
    #     "/Users/vamsikeshwaran/animal/farm-6b70f-firebase-adminsdk-dnr8b-6fb03f1a9b.json")
    # firebase_admin.initialize_app(cred, {
    #     'databaseURL': 'https://farm-6b70f-default-rtdb.firebaseio.com/'
    # })

    # ref = db.reference('/')

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
                    try:
                        label_name = model.names[label]
                        if label_name == "sheep":
                            print("Detected: pig")
                            # print(str(datetime.now()))
                            data = {
                                'is_detected': "true"
                            }
                            # ref.child('-Nv7k9_XUoJtnF6P5YQB').update(data)
                            # datas = {
                            #     'Animal_detected': "pig",
                            #     'time_stamp': str(datetime.now())
                            # }

                            # ref.child('-NvCCQB4XAL572083hD7').update(datas)

                        else:
                            print(f'Detected: {label_name}')
                            # print(str(datetime.now()))
                            data = {
                                'is_detected': "true"
                            }

                            # ref.child('-Nv7k9_XUoJtnF6P5YQB').update(data)
                            # datas = {
                            #     'Animal_detected': str(label_name),
                            #     'time_stamp': str(datetime.now())
                            # }
                            # ref.child('-NvCCQB4XAL572083hD7').update(datas)

                    except Exception as e:
                        print("Exception occurred:", e)

                else:
                    print("Confidence lower than threshold")
        else:
            data = {
                'is_detected': "false"
            }

            # ref.child('-Nv7k9_XUoJtnF6P5YQB').update(data)

        frame_np = np.array(frame)

        for det in detections:
            bbox = det[:4].cpu().numpy().astype(
                int)
            cv2.rectangle(frame_np, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]), (0, 255, 0), 2)

        return frame_np

    cap = cv2.VideoCapture('test3.mp4')

    while True:
        ret, frame = cap.read()

        processed_frame = detect_animals(frame)

        cv2.imshow('Animal Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return "Working"
    # return jsonify(response), 200


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000, debug=True)
    animal_detection()