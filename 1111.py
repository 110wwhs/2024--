import speech_recognition as sr

def recognize_speech():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("음성 입력 중...")
            audio = recognizer.listen(source)

            try:
                text2 = recognizer.recognize_google(audio, language='ko-KR')
                print("음성 입력 결과:", text2)
                return text2
            except sr.UnknownValueError:
                print("음성을 인식할 수 없습니다.")
                return None
            except sr.RequestError:
                print("음성 인식 서비스에 접근할 수 없습니다.")
                return None
    except OSError:
        print("기본 입력 장치를 찾을 수 없습니다. 마이크가 연결되어 있는지 확인해주세요.")
        return None

# 함수 호출 예제
recognize_speech()
