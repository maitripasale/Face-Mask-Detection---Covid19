# uvicorn main:app --host 0.0.0.0
# http://192.168.43.241:8000/docs
# http://localhost:8000/docs


from plyer import notification
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from plyer import notification
from fastapi import FastAPI, File, UploadFile, Form
import uuid
from playsound import playsound

app = FastAPI()

# construct the argument parser and parse the arguments
# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model = load_model("mask_detector.model")


def mask_image(image):
    # load the input image from disk, clone it, and grab the image spatial
    # dimensions
    file_to_process = image
    image = cv2.imread(image)

    orig = image.copy()
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = model.predict(face)[0]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            if label == "No Mask":
                notification.notify(
                    title="***No Mask Detected***",
                    message="Wear Mask to stay safe! ",
                    app_icon="images/1.ico",  # ico file should be downloaded
                    timeout=1
                )
                playsound("no_mask.wav")
                cv2.imshow("Output", image)
                cv2.waitKey(0)
                new_image = "results/" + "NO_MASK_" + str(uuid.uuid4()) + ".jpg"
                cv2.imwrite(new_image, image)
            else:
                new_image = "results/" + "MASK_" + str(uuid.uuid4()) + ".jpg"
                cv2.imwrite(new_image, image)

    os.remove(file_to_process)
    result_data = {
        "label": label,
        "image": new_image
    }
    return result_data


def _save(file):
    file_name = file.filename
    with open(file_name, 'wb') as f:
        f.write(file.file.read())
    return file_name


def generate_pdf(label, image):
    """code to generate pdf file"""
    pdf_file_name = "pdf_files/" + str(uuid.uuid4()) + ".pdf"
    print("*******************************")
    print("************CHALLAN************")
    print("FILE: ", pdf_file_name)
    print("DETECTION: ", label)
    print("IMAGE:", image)
    print("FINE: ", "RS.500")
    print("*******************************")
    print("*******************************")

    # os.remove(image)


@app.post("/predict/")
def predict_mask(file: UploadFile = File(...)):
    file_name = _save(file)
    result = mask_image(file_name)
    """result can be Mask or No Mask """
    label = result["label"]
    image_to_put_in_pdf = result["image"]
    if label == "No Mask":
        """generate a pdf having the label and image_to_put_in_pdf"""
        generate_pdf(label, image_to_put_in_pdf)
    return label
