import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from gtts import gTTS
import threading
from playsound import playsound
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import ImageFont, ImageDraw, Image

# MediaPipe 손 솔루션 불러오기
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

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
y_test = np.array([label_map[label] for label in y_test])
num_classes = len(label_map)

# CNN 모델 구축
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 카메라 설정
cap = cv2.VideoCapture(0)

# 폰트 경로 설정
font_path = "D2Coding-Ver1.3.2-20180524.ttf"
font = ImageFont.truetype(font_path, 32)

# 수어를 한국어로 번역하는 기능
def translate_sign_to_korean(sign):
    try:
        korean_translation = mapping_df[mapping_df["파일명"] == sign]["한국어"].values[0]
        print("한국어 번역:", korean_translation)
        return korean_translation
    except IndexError:
        print("해당 수어에 대한 한국어 번역을 찾을 수 없습니다.")
        return "유사한 단어 없음"

# TTS를 비동기적으로 실행하는 함수
def play_translation(text):
    tts = gTTS(text=text, lang='ko')
    tts.save("temp.mp3")
    playsound("temp.mp3")
    os.remove("temp.mp3")

# 카메라로부터 수어를 인식하고 번역하는 함수
def capture_sign_to_korean():
    while True:
        # 카메라 프레임 확인
        if not cap.isOpened():
            print("카메라가 열리지 않았습니다.")
            break

        ret, frame = cap.read()
        if not ret:
            print("카메라로부터 프레임을 읽을 수 없습니다.")
            continue

        # 손 인식
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 손 인식 결과 확인
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 수어 번역
                landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
                input_image = landmarks_to_image(landmarks).reshape(-1, 128, 128, 1)
                prediction = model.predict(input_image)
                predicted_label = np.argmax(prediction, axis=1)[0]
                closest_file = [label for label, idx in label_map.items() if idx == predicted_label][0]
                translated_text = translate_sign_to_korean(closest_file) if closest_file else "유사한 단어 없음"

                # 한국어 번역을 이미지에 추가
                pil_image = Image.fromarray(image)
                draw = ImageDraw.Draw(pil_image)
                draw.text((10, 10), translated_text, font=font, fill=(255, 255, 255, 255))
                image = np.array(pil_image)

                # TTS 실행
                if translated_text != "유사한 단어 없음":
                    threading.Thread(target=play_translation, args=(translated_text,)).start()

        # 화면에 이미지 표시
        cv2.imshow('Korean Sign Language Recognition', image)

        # 종료 키 확인
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 비디오 파일에서 랜드마크 추출 함수 (최대 100개 비디오 파일로 제한)
def extract_landmarks_from_videos(video_dir):
    video_files = [os.path.join(video_dir, file) for file in os.listdir(video_dir) if file.endswith((".avi", ".mov", ".mts", ".mp4"))]
    video_files = video_files[:100]  # 최대 100개 비디오 파일만 로딩
    if not video_files:
        print("비디오 파일을 찾을 수 없습니다.")
        return

    all_landmarks = []

    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        file_landmarks = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
                    file_landmarks.append(landmarks)

        cap.release()

        if file_landmarks:
            avg_landmarks = np.mean(file_landmarks, axis=0)
            all_landmarks.append({"파일명": os.path.basename(video_file), "랜드마크": avg_landmarks.tolist()})

    landmarks_df = pd.DataFrame(all_landmarks)
    landmarks_df.to_csv(landmarks_file, index=False, encoding='euc-kr')
    print("랜드마크 파일 저장 완료:", landmarks_file)

# 메인 함수
def main():
    video_dir = "수어 데이터셋"

    # 랜드마크 추출 및 저장
    extract_landmarks_from_videos(video_dir)

    # 카메라로부터 수어를 인식하고 번역하는 함수 호출
    print("카메라를 통해 수어를 인식합니다. 'q' 키를 눌러 종료하세요.")
    capture_sign_to_korean()

if __name__ == "__main__":
    main()
    cv2.imshow('Korean Sign Language Translator', translated_image)

    # 종료 키 확인
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
