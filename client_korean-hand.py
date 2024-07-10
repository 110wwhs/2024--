import os
import cv2
import threading
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import speech_recognition as sr

# 텍스트 파일 경로
txt_file = "KETI-2018-SL-Annotation-v1.txt"

# 수어 데이터셋 디렉토리
video_dir = "수어 데이터셋"

# 전역 변수로 video_mapping 초기화
video_mapping = {}

# 텍스트 파일에서 매핑을 불러오는 함수
def load_video_mapping(txt_file):
    try:
        video_mapping = {}
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    video_file, korean_word = parts
                    if video_file.lower() == "파일명":
                        continue
                    try:
                        base_index = int(video_file.split('_')[-1])
                        for i in range(0, 3145, 419):
                            repeated_video_file = f"KETI_SL_{base_index + i:010d}"
                            video_mapping[korean_word.strip()] = repeated_video_file.strip()
                    except ValueError:
                        print(f"Invalid video file format: {video_file}")
                        continue
        return video_mapping
    except UnicodeDecodeError:
        print("텍스트 파일을 읽을 때 인코딩 오류가 발생했습니다. 인코딩을 확인하세요.")
        return {}

# 한국어 텍스트를 입력받아 수어 영상 파일명 반환
def get_sign_video(text, video_mapping):
    return video_mapping.get(text.strip(), None)

# 수어 영상을 재생하는 함수
def play_sign_video(video_file):
    cap = cv2.VideoCapture(video_file)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV 이미지를 PIL 이미지로 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)
        root.update_idletasks()
        if not recognizing:
            break
    cap.release()

# 음성 인식을 통해 한국어 텍스트 입력받기
def recognize_speech():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print("음성 입력 중...")
            audio = recognizer.listen(source)
            text2 = recognizer.recognize_google(audio, language= 'ko-KR')
            print(text2)

            try:
                text = recognizer.recognize_google(audio, language='ko-KR')
                print("음성 입력:", text)
                return text
            except sr.UnknownValueError:
                print("음성을 인식할 수 없습니다.")
                return None
            except sr.RequestError:
                print("음성 인식 서비스에 접근할 수 없습니다.")
                return None
    except OSError:
        print("기본 입력 장치를 찾을 수 없습니다. 마이크가 연결되어 있는지 확인해주세요.")
        return None

def start_speech_recognition():
    global recognizing
    recognizing = True
    threading.Thread(target=recognize_and_play_video).start()

def stop_speech_recognition(event):
    global recognizing
    recognizing = False

def recognize_and_play_video():
    global recognizing
    while recognizing:
        text = recognize_speech()
        if not text:
            continue
        text_var.set("음성 인식 결과: " + text)
        video_file = get_sign_video(text, video_mapping)
        if video_file:
            video_file_path = os.path.join(video_dir, video_file)
            if os.path.exists(video_file_path):
                threading.Thread(target=play_sign_video, args=(video_file_path,)).start()
            else:
                text_var.set("수어 영상 파일을 찾을 수 없습니다: " + video_file_path)
        else:
            text_var.set("해당하는 수어 영상을 찾을 수 없습니다: " + text)

def start_text_input():
    global recognizing
    recognizing = False
    text = text_entry.get()
    text_var.set("텍스트 입력 결과: " + text)
    video_file = get_sign_video(text, video_mapping)
    if video_file:
        video_file_path = os.path.join(video_dir, video_file)
        if os.path.exists(video_file_path):
            threading.Thread(target=play_sign_video, args=(video_file_path,)).start()
        else:
            text_var.set("수어 영상 파일을 찾을 수 없습니다: " + video_file_path)
    else:
        text_var.set("해당하는 수어 영상을 찾을 수 없습니다: " + text)

def on_closing():
    global root
    if messagebox.askokcancel("Quit", "프로그램을 종료하시겠습니까?"):
        global recognizing
        recognizing = False
        root.destroy()

# GUI 설정
root = tk.Tk()
root.title("한국어 음성/텍스트 인식 및 수어 영상 재생")
root.geometry("800x600")

recognizing = False

# 텍스트 파일에서 video_mapping 초기화
video_mapping = load_video_mapping(txt_file)

instructions = tk.Label(root, text="스페이스바를 누르고 있으면 음성 인식이 시작됩니다.")
instructions.pack(pady=10)

voice_btn = tk.Button(root, text="음성 인식 시작", command=start_speech_recognition)        
voice_btn.pack(pady=10)

text_entry = tk.Entry(root, width=50)
text_entry.pack(pady=10)

text_btn = tk.Button(root, text="텍스트 입력 시작", command=start_text_input)
text_btn.pack(pady=10)

text_var = tk.StringVar()
text_label = tk.Label(root, textvariable=text_var)
text_label.pack(pady=10)

video_label = tk.Label(root)
video_label.pack(pady=10)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.bind('<space>', lambda event: start_speech_recognition())
root.bind('<KeyRelease-space>', stop_speech_recognition)
root.mainloop()
