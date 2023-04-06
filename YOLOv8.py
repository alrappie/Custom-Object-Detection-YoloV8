import cv2
from ultralytics import YOLO

PATH = './Video/'

cap = cv2.VideoCapture('./data/images/train/00000000.jpg')
wCam, hCam = 1280,720
# cap.set(3,wCam)
# cap.set(4,hCam)
# H, W, _ = frame.shape

# video_path_out = '{}_out.mp4'.format(PATH)
# out = cv2.VideoWriter(video_path_out,cv2.VideoWriter_fourcc(*'MP4V'),int(cap.get(cv2.CAP_PROP_FPS)),(W,H))

model_path = r'ultralytics\runs\detect\train\weights\best.pt'

model = YOLO(model_path)

threshold = 0.5

class_name_dict = {0: 'seblak'}

while True:
    ret, frame = cap.read()

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # out.write(frame)
    # ret, frame = cap.read()
    
    cv2.imshow('Image',frame)
    cv2.waitKey(1)
# cap.release()
# out.release()
# cv2.destroyAllWindows()