from kafka import KafkaConsumer
from flask import Flask, Response, render_template
import cv2
import numpy as np
from SparkStreaming import *
from Predictor import Predictor

topic = 'tutorial'

consumer = KafkaConsumer(topic, bootstrap_servers=['localhost:9092'])

app = Flask(__name__)


@app.route('/', methods=['GET'])
def video_feed():
    return Response(get_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_video_stream():
    for msg in consumer:
        processed_image = process_image(msg.value)
        yield (b'--frame\r\n' b'Content-Type: image/jpg\r\n\r\n' + processed_image + b'\r\n\r\n')

def process_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    fake_img = predictor.generate(image_data)

    ret, buffer = cv2.imencode('.jpg', fake_img)
    return buffer.tobytes()


if __name__ == '__main__':
    predictor = Predictor(model_path='generator_hayao.pth')

    app.run(debug=True)