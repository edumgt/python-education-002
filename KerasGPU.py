import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ✅ GPU 확인
print("사용 가능한 GPU:", tf.config.list_physical_devices('GPU'))

# ✅ 데이터 로드 (MNIST 숫자 이미지)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# ✅ 데이터 전처리 (GPU 효율을 위해 float16 사용)
x_train = x_train.reshape(-1, 28*28).astype("float16") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float16") / 255.0

# ✅ 혼합 정밀도 정책 적용 (float16 + float32)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# ✅ 모델 정의
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax', dtype='float32')  # 출력은 float32로 설정!
])

# ✅ 모델 컴파일
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ 모델 학습 (자동으로 GPU 사용)
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# ✅ 테스트 평가
model.evaluate(x_test, y_test)
