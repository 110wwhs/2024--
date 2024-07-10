import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# MediaPipe 손 솔루션 불러오기
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# 비디오 파일에서 랜드마크 추출 함수
def extract_landmarks_from_video(video_file):
    cap = cv2.VideoCapture(video_file)
    file_landmarks = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1초 간격으로 프레임 처리 (더 빠르게 하기 위해)
        if frame_count % int(cap.get(cv2.CAP_PROP_FPS) * 1) == 0:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
                    file_landmarks.append(landmarks)

        frame_count += 1

    cap.release()

    if file_landmarks:
        avg_landmarks = np.mean(file_landmarks, axis=0)
        return {"파일명": os.path.basename(video_file), "랜드마크": avg_landmarks.tolist()}
    return None

# 비디오 디렉토리의 모든 비디오 파일 처리 함수
def process_all_videos(video_dir, output_file):
    video_files = [os.path.join(video_dir, file) for file in os.listdir(video_dir) if file.endswith((".avi", ".mov", ".mts", ".mp4"))]
    if not video_files:
        print("비디오 파일을 찾을 수 없습니다.")
        return

    all_landmarks = []
    for video_file in video_files:
        print(f"Processing {video_file}")
        landmarks = extract_landmarks_from_video(video_file)
        if landmarks:
            all_landmarks.append(landmarks)

    if all_landmarks:
        landmarks_df = pd.DataFrame(all_landmarks)
        landmarks_df.to_csv(output_file, index=False, encoding='euc-kr')
        print("랜드마크 파일 저장 완료:", output_file)
    else:
        print("랜드마크를 추출하지 못했습니다.")

# 메인 함수
def main():
    video_dir = "수어 데이터셋"
    output_file = "landmarks_mapping.csv"
    process_all_videos(video_dir, output_file)

if __name__ == "__main__":
    main()
