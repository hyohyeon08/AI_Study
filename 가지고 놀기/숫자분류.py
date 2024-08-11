import tensorflow as tf
from keras.datasets.mnist import load_data
from keras.models import Sequential
from keras import models, layers
from keras.layers import Dense, Input, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image

#mnist 데이터 불러오기
(x_train_full, y_train_full), (x_test, y_test) = load_data(path='mnist.npz')
x_train, x_val, y_trian, y_val = train_test_split(x_train_full, y_train_full, test_size=0.3)

#데이터 전처리 작업
x_train = x_train / 255.
x_val = x_val / 255.
x_test = x_test / 255.

y_train = to_categorical(y_trian)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

#신경망 만들기
model = models.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(), 
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(10, activation='softmax')
])

#모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#모델 학습
history = model.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_val, y_val))

#모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy : {test_acc:.4f}")
print(f"Test loss : {test_loss:.4f}")


#내가 가져온 사진으로 예측하기
img_path = 'test_1.png'
img = image.load_img(img_path, target_size=(28,28), color_mode='grayscale')
img_array = image.img_to_array(img) / 255. #이미지를 0 ~ 1로 정규화
img_array = np.expand_dims(img_array, axis=0) #배치 차원 추가 (모델의 입력 형식에 맞게)

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

img_array = np.squeeze(img_array) #이미지 시각화를 위한 전처리 -> 배치 차원 다시 제거 

print(f"Predicted class : {predicted_class}")

#이미지 출력
plt.imshow(img_array, cmap='gray')
plt.title(f"Predicted : {predicted_class}")
plt.show()

#