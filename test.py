from SparkStreaming import *

if __name__=='__main__':
    spark_connection = create_spark_connection()
    bc_model_state = load_model(spark_connection, 'generator_hayao.pth')
    img = cv2.imread('1.png')
    predict_udf(bc_model_state, img)