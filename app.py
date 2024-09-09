from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model('keras_model.h5', compile=False)

# Load the labels
class_names = open('labels.txt', 'r').readlines()

st.header('이수지 vs 김고은 vs 싸이')


# Create the array of the right shape to feed into the keras model0
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

uploaded_file = st.file_uploader("파일을 업로드하세요", type=["jpg", "jpeg", "png"])
#img_file_buffer = st.camera_input("정중앙에 사물을 위치하고 사진찍기 버튼을 누르세요")

if uploaded_file is not None:
    # Replace this with the path to your image
    image = Image.open(uploaded_file).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    # 모델이 학습했을 때 Nomalize 한 방식대로 이미지를 Nomalize 
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    # 빈 ARRAY에 전처리를 완료한 이미지를 복사
    data[0] = normalized_image_array

    # Predicts the model
    # h5 모델에 예측 의뢰 
    prediction = model.predict(data)

    # 높은 신뢰도가 나온 인덱의 인덱스 자리를 저장
    index = np.argmax(prediction)

    # labels.txt 파일에서 가져온 값을 index로 호출
    class_name = class_names[index]
    # 예측 결과에서 신뢰도를 꺼내 옵니다  
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    st.write('Class:', class_name[2:], end="")
    st.write('Confidence score:', confidence_score)
