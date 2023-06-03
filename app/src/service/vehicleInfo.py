import uuid
import json
import requests
from fastapi import FastAPI, UploadFile
from datetime import datetime, time
from typing import List, Optional
from pydantic import BaseModel
from pydantic.types import UUID4, constr
import cv2
import numpy as np
from src.service.local_utils import detect_lp
from os.path import splitext, basename
from keras.models import model_from_json
import glob
import pytesseract
from PIL import Image
from src.model.vehicle import extractedInfo
from roboflow import Roboflow
import re


def countVehicles(name):
    image = Image.open(name)
    image = image.resize((450, 250))
    image_arr = np.array(image)
    grey = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
    Image.fromarray(grey)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    Image.fromarray(blur)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    dilated = cv2.dilate(blur, np.ones((3, 3)))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    Image.fromarray(closing)
    car_cascade_src = '/Users/binarycoder/Documents/mumbaiHacks/app/src/ml/cars.xml'
    car_cascade = cv2.CascadeClassifier(car_cascade_src)
    cars = car_cascade.detectMultiScale(closing, 1.1, 1)
    cnt = 0
    for (x, y, w, h) in cars:
        cv2.rectangle(image_arr, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cnt += 1
    return cnt


def load_model(path):
    try:
        path = splitext(path)[0]
        print(path)
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights(
            '/Users/binarycoder/Documents/mumbaiHacks/app/src/ml/wpod-net.h5')
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)


def preprocess_image(image_path, resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224, 224))
    return img


def get_plate(image_path, wpod_net, Dmax=608, Dmin=256):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    print(image_path)
    _, LpImg, _, cor = detect_lp(
        wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor

# def countCar():
#     class_name = []
#     with open('../ml/classes.txt', 'r') as f:
#         class_name = [cname.strip() for cname in f.readlines()]
#     # print(class_name)
#     net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
#     model = cv2.dnn_Detectpip ionModel(net)
#     model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


def extractCarColor(name):
    pass


def extractvehicleInfo(name):
    rf = Roboflow(api_key="QMdargV4HeLKYFatDOkB")
    project = rf.workspace().project("license-plate-recognition-rxg4e")
    model = project.version(4).model

    ans = model.predict(
        name, confidence=40, overlap=30).json()
    print("ans:", ans['predictions'][0]['x'])
    print("ans:", ans['predictions'][0]['width'])
    prediction = ans['predictions'][0]
    roi_x = int(prediction['x'] - prediction['width'] / 2)
    roi_y = int(prediction['y'] - prediction['height'] / 2)
    roi_width = int(prediction['width'])
    roi_height = int(prediction['height'])
    image = cv2.imread(name)
    cropped_image = image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
    cv2.imwrite(name, cropped_image)
    number1 = validate_vehicle_number(
    pytesseract.image_to_string(cropped_image))
    number2 = validate_vehicle_number(ocr_space_api(name))
    print("number1: ", number1)
    print("number2: ", number2)
    return number1 if len(number1) > len(number2) else number2


def ocr_space_api(image_path, language="eng", is_table=False):
    api_key = "K83555174088957"
    # OCR.Space API endpoint
    api_url = "https://api.ocr.space/parse/image"

    # Request headers
    headers = {
        "apikey": api_key
    }

    # Request payload
    payload = {
        "language": language,
        "isTable": str(is_table),
        "OCREngine": "2"
    }

    # Read the image file as binary data
    with open(image_path, "rb") as file:
        image_data = file.read()

    # Request data
    files = {
        "filename": (image_path, image_data)
    }

    # Send POST request to OCR.Space API
    response = requests.post(api_url, headers=headers,
                             data=payload, files=files)

    # Get the JSON response
    json_response = response.json()
    parsed_text = json_response["ParsedResults"][0]["ParsedText"]
    return parsed_text
    # Check if the request was successful
    # if response.status_code == 200 and json_response["IsErroredOnProcessing"] == "false":
    #     # Extract the parsed text from the response

    # else:
    #     # If the request was not successful, return None
    #     return None


def extractVehicleInfo(name: UploadFile):
    name = '/Users/binarycoder/Documents/mumbaiHacks/app/src/img/Cars9.jpeg'
    cnt = countVehicles(name)
    if cnt > 0:
        wpod_net_path = "/Users/binarycoder/Documents/mumbaiHacks/app/src/ml/wpod-net.json"
        wpod_net = load_model(wpod_net_path)
        img_path = glob.glob(name)
        print("ty", img_path)
        LpImg, cor = get_plate(name, wpod_net)
        print("Coordinate of plate(s) in image: \n", cor)
        plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

        number = pytesseract.image_to_string(plate_image)
        print("number:", number)
        number = number.strip()
        print("number2:", number)
        number = validate_vehicle_number(number)
        print("number1:", number)
        return {
            "isVehicleExit": True,
            "vehicleNumber": number,
            "color": ""
        }
    return {
        "isVehicleExit": True,
        "vehicleNumber": number,
        "color": None
    }


def validate_vehicle_number(vehicle_number):
    pattern = r'^[A-Z]{1,2}\s?\d{1,2}\s?[A-Z]{1,2}\s?\d{1,4}$'
    return re.findall(pattern, vehicle_number)
