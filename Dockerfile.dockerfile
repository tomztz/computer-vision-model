FROM python:3.7.0

WORKDIR /computer-vision-model/tensorflow-yolov4-tflite .

COPY . .

RUN pip install -r requirements.txt

CMD [ "python", "detect_video.py" ]