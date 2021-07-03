import requests
import time
import subprocess

url = "http://localhost:8000/predict/"

file_name = "images/image.jpg"


while True:
    #subprocess.call(["fswebcam", "-r", "640×480", file_name])
    """above code captures images"""
    with open(file_name, 'rb') as f:
        data = f.read()

    files = [
        ('file', (file_name, data, 'image/jpeg'))
    ]

    response = requests.request("POST", url, files=files)

    print(response.text)
    time.sleep(10)