# To generate grey image for dataset
import cv2
import numpy as np


def thermal_color_map(value):
    colors = np.array([
        (0, 0, 0),
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 255, 0),
        (255, 0, 0),
        (255, 255, 255)
    ], dtype=np.uint8)

    thresholds = np.array([36, 73, 109, 146, 182, 219])

    index = np.argmax(value < thresholds)

    return colors[index]


video_path = 'D:\My Workspace\Projects\Flask - Framework\\animal_detection\\rough_assets\Elephant.mp4'
output_path = 'output_video.mp4'

cap = cv2.VideoCapture(video_path)


fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    thermal_image = np.apply_along_axis(
        thermal_color_map, -1, gray[..., np.newaxis])

    out.write(thermal_image)

    cv2.imshow('Thermal Image', thermal_image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()