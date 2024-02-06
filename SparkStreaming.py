from pyspark.sql import SparkSession
import numpy as np
import cv2
import logging
from Generator import Generator
import torch

def create_spark_connection():
    s_conn = None
    try:
        # s_conn = SparkSession.builder \
        #     .appName('SparkDataStreaming') \
        #     .getOrCreate()
        #     # .config('spark.jars.packages', "org.apache.spark:spark-sql-kafka-0-10_2.13:3.4.1") \
        s_conn = SparkSession.builder.master('local[*]').appName("model_training").getOrCreate()
        s_conn.sparkContext.setLogLevel("ERROR")
        logging.info("Spark connection created successfully!")
    except Exception as e:
        logging.error(f"Couldn't create the spark session due to exception {e}")

    return s_conn


def connect_to_kafka(spark_conn):
    spark_df = None
    try:
        spark_df = spark_conn.readStream \
            .format('kafka') \
            .option('kafka.bootstrap.servers', 'localhost:9092') \
            .option('subscribe', 'tutorial') \
            .option('startingOffsets', 'earliest') \
            .load()
        logging.info("kafka dataframe created successfully")
    except Exception as e:
        logging.warning(f"kafka dataframe could not be created because: {e}")

    return spark_df


def load_model(s_conn, model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    Gen = Generator('Hayao')
    Gen.load_state_dict(checkpoint['model_state_dict'])
    Gen.eval()
    bc_model_state = s_conn.sparkContext.broadcast(Gen.state_dict())
    return bc_model_state


def input_transform(img):
    image = img[:,:,::-1]
    image = cv2.resize(image, (image.shape[1] // 4*4, image.shape[0] // 4*4))
    image = image.astype(np.float32)
    image = normalize_input(image)
    image = image.transpose(2, 0, 1)
    return torch.tensor(image)


def normalize_input(images):
    '''[0, 255] -> [-1, 1]'''
    return images / 127.5 - 1.0

def denormalize_input(images, dtype=None):
    '''[-1, 1] -> [0, 255]'''
    images = images * 127.5 + 127.5
    if dtype is not None:
        if isinstance(images, torch.Tensor):
            images = images.type(dtype)
        else:
            images = images.astype(dtype)
    return images

def predict_udf(bc_model_state, image_data):
    model_state = bc_model_state.value
    model = Generator('Hayao')
    model.load_state_dict(model_state)
    model.eval()
    
    # Chuyển đổi dữ liệu hình ảnh sang định dạng PyTorch Tensor
    image_tensor = input_transform(image_data)[None,:,:,:]

    # Thực hiện dự đoán
    with torch.no_grad():
        fake_img = model(image_tensor)
        fake_img = fake_img.detach().cpu().numpy()
        fake_img = fake_img.transpose(0, 2, 3, 1)
        fake_img = denormalize_input(fake_img, dtype=np.int16)
    cv2.imwrite('result/1.jpg', fake_img[0][..., ::-1])
    cv2.imshow('',fake_img[0][..., ::-1])
    cv2.waitKey(0)
    return fake_img[0][..., ::-1]