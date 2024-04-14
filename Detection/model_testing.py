import cv2
import os
import numpy as np


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


folder_path = "D:\My Workspace\Projects\Flask - Framework\\animal_detection\\animals\\animals\\aadetection"
animal_images = {}
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg") :
        animal_name = os.path.splitext(filename)[0]
        image = cv2.imread(os.path.join(folder_path, filename))
        animal_images[animal_name] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    min_mse = float('inf')
    min_animal = None

    for animal_name, animal_image in animal_images.items():

        animal_image_resized = cv2.resize(
            animal_image, (frame_gray.shape[1], frame_gray.shape[0]))

        error = mse(frame_gray, animal_image_resized)

        if error < min_mse:
            min_mse = error
            min_animal = animal_name

    if min_mse < 10000 and min_animal is not None:
        cv2.rectangle(
            frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)
        cv2.putText(frame, min_animal, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()