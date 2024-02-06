from kafka import KafkaProducer
from json import dumps
from time import sleep
import cv2
import sys

topic = 'tutorial'

def publish_video(video_file):
    producer = KafkaProducer(bootstrap_servers='localhost:9092')

    video = cv2.VideoCapture(video_file)
    while video.isOpened():
        success, frame = video.read()
        if not success:
            print("Reached end of video.")
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        producer.send(topic, buffer.tobytes())

        sleep(0.2)
    video.release()


def publish_camera():
    producer = KafkaProducer(bootstrap_servers='localhost:9092')

    camera = cv2.VideoCapture(0)
    try:
        while True:
            success, frame = camera.read()

            ret, buffer = cv2.imencode('.jpg', frame)
            producer.send(topic, buffer.tobytes())

            sleep(0.2)
    except:
        print('\nexit')
        sys.exit(1)

if __name__ == '__main__':
    a = 2
    if a > 1:
        print('publishing feed!')
        video_path = "D:\\UIT\\K5\\BigData\\FINAL_PROJECT\\4k-video.mp4"
        publish_video(video_path)
    else:
        print('publishing feed!')
        publish_camera()