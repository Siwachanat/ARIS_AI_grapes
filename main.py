# imports 
import cv2 as cv 
import numpy as np
import time
import AiPhile 
import re
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow.lite as tflite
from PIL import Image
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 320

def load_labels(label_path):
    r"""Returns a list of labels"""
    with open(label_path) as f:
        labels = {}
        for line in f.readlines():
            m = re.match(r"(\d+)\s+(\w+)", line.strip())
            labels[int(m.group(1))] = m.group(2)
        return labels
def load_model(model_path):
    r"""Load TFLite model, returns a Interpreter instance."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def getAvailableCameraIds(max_to_test):
    available_ids = []
    for i in range(max_to_test):
        temp_camera = cv.VideoCapture(i)
        if temp_camera.isOpened():
            temp_camera.release()
            print("found camera with id {}".format(i))
            available_ids.append(i)
    return available_ids

def detectQRcode(image):
    # convert the color image to gray scale image
    Gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # create QR code object
    objectQRcode = pyzbar.pyzbar.decode(Gray)
    for obDecoded in objectQRcode: 
        x, y, w, h =obDecoded.rect
        cv.rectangle(image, (x,y), (x+w, y+h), ORANGE, 4)
        points = obDecoded.polygon
        hull = points
  
        return hull


#set model

model_path = 'aris-object-detection-tensorflow-lite-float32-model.lite'
label_path = 'label.txt'

#getAvailableCameraIds(10)

cap = cv.VideoCapture(0,cv.CAP_DSHOW)
#cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FOURCC,  1196444237)
cap.set(3,640)
cap.set(4,640)
cap.set(cv.CAP_PROP_FPS, 2)
ret, frame = cap.read()
image_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
image_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
print(ret)
print(frame)


frame_counter =0
starting_time =time.time()

interpreter = load_model(model_path)
labels = load_labels(label_path)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
width = input_details[0]['shape'][2]
height = input_details[0]['shape'][1]

# Get input index
input_index = input_details[0]['index']

while True:
    ret, frame = cap.read()

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_resized = cv.resize(frame_rgb, (width, height))
    frame_resized = frame_resized.astype(np.float32)
    frame_resized /= 255.
    input_data = np.expand_dims(frame_resized, axis=0)
    # set frame as input tensors
    interpreter.set_tensor(input_details[0]['index'], input_data)
    # perform inference
    interpreter.invoke()
    # Get output tensor
    output_details = interpreter.get_output_details()
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    #print("-->",scores )
    for i in range(len(scores)):
        if ((scores[i] > 0.7) and (scores[i] <= 1.0)):
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * image_height)))
            xmin = int(max(1, (boxes[i][1] * image_width)))
            ymax = int(min(image_height, (boxes[i][2] * image_height)))
            xmax = int(min(image_width, (boxes[i][3] * image_width)))

            cv.rectangle(frame, (xmin, ymin),
                          (xmax, ymax), (10, 255, 0), 4)
            print((xmin+xmax)//2 , (ymin+ymax)//2)
            # Draw label
            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))
            labelSize, baseLine = cv.getTextSize(
                label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # Make sure not to draw label too close to top of window
            label_ymin = max(ymin, labelSize[1] + 10)
            cv.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (
                xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (xmin, label_ymin - 7),
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)




    cv.imshow("image", frame)

    # Press 'q' to quit
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

# keep looping until the 'q' key is pressed
