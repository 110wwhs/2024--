import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# 데이터 파일 경로
csv_file = "KETI-2018-SL-Annotation-v1.csv"
landmarks_file = "landmarks_mapping.csv"

# 데이터 파일 불러오기
try:
    mapping_df = pd.read_csv(csv_file, encoding='euc-kr')
    landmarks_df = pd.read_csv(landmarks_file, encoding='euc-kr')
    print("데이터 파일 불러오기 성공:", csv_file, landmarks_file)
except Exception as e:
    print("데이터 파일 불러오기 실패:", e)
    exit()

# 랜드마크 데이터를 이미지로 변환하는 함수
def landmarks_to_image(landmarks, image_size=128):
    image = np.zeros((image_size, image_size), dtype=np.uint8)
    for i in range(0, len(landmarks), 3):
        x = int(landmarks[i] * image_size)
        y = int(landmarks[i + 1] * image_size)
        if 0 <= x < image_size and 0 <= y < image_size:
            image[y, x] = 255
    return image

# 데이터셋 준비
images = []
labels = []
for index, row in landmarks_df.iterrows():
    landmarks = np.array(eval(row["랜드마크"]))
    image = landmarks_to_image(landmarks)
    images.append(image)
    labels.append(row["파일명"])

images = np.array(images).reshape(-1, 128, 128, 1)
labels = np.array(labels)

# 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 라벨 인코딩
label_map = {label: idx for idx, label in enumerate(np.unique(y_train))}
y_train = np.array([label_map[label] for label in y_train])

# 라벨 매핑 체크 및 수정
def map_labels(labels, label_map):
    mapped_labels = []
    missing_labels = set()
    for label in labels:
        if label in label_map:
            mapped_labels.append(label_map[label])
        else:
            missing_labels.add(label)
    if missing_labels:
        print(f"Missing labels: {missing_labels}")
        for label in missing_labels:
            label_map[label] = len(label_map)
            mapped_labels.append(label_map[label])
    return np.array(mapped_labels)

y_test = map_labels(y_test, label_map)
num_classes = len(label_map)

# 초기 설정
best_accuracy = 0.0
epochs = 0

while best_accuracy < 0.7:
    # CNN 모델 구축
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 모델 저장 경로
    model_path = "sign_language_model_temp.keras"

    # 모델 체크포인트 설정
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # 모델 훈련
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[checkpoint], verbose=1)

    # 학습된 모델의 최고 정확도 확인
    max_val_accuracy = max(history.history['val_accuracy'])

    # 현재 최고 정확도와 비교하여 더 높으면 저장 및 업데이트
    if max_val_accuracy > best_accuracy:
        best_accuracy = max_val_accuracy
        os.rename(model_path, "sign_language_model.keras")
        print(f"현재 최고 정확도: {best_accuracy:.4f}")

    epochs += 10

print(f"모델 학습 완료. 최종 최고 정확도: {best_accuracy:.4f}, 총 Epochs: {epochs}")
