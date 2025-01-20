import cv2
import glob
import time

image_paths = sorted(glob.glob('runs/detect/exp/*.jpg'))

cv2.namedWindow('YOLO Inference', cv2.WINDOW_NORMAL)

for image_path in image_paths:
    image = cv2.imread(image_path)

    cv2.imshow('YOLO Inference', image)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    time.sleep(0.1)

cv2.destroyAllWindows()
