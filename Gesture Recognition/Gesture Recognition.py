import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from time import time


model_path = 'gesture_recognizer.task'  # mention path to task file, which is downloaded from MediaPipe website.

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path = model_path),
    running_mode=VisionRunningMode.IMAGE)

cap = cv2.VideoCapture(0)
with GestureRecognizer.create_from_options(options) as recognizer:
    while cap.isOpened():
        ret, frame = cap.read()
    
        if not ret:
            continue
        # flipping the image
        frame = cv2.flip(frame, 1)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    
        res = recognizer.recognize(mp_image)

        text=''
        try:
            text=res.gestures[0][0].category_name
        except:
            text="None"
        print(text)
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(frame, text,(50,50), font,1, (0,0,255),2,cv2.LINE_AA )

        cv2.imshow('frame',frame)
        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()