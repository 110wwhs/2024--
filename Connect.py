from konlpy.tag import Komoran
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# KoNLPy의 형태소 분석기인 Komoran 초기화
komoran = Komoran()

# 사전 훈련된 Word2Vec 모델 로드
word2vec_model = Word2Vec.load('path_to_pretrained_word2vec_model')

def preprocess_sentence(sentence):
    # 문장을 단어로 토큰화하고 형태소 분석 수행
    tokens = komoran.morphs(sentence)
    return tokens

def get_sentence_vector(tokens):
    # 각 단어의 임베딩 벡터를 가져와서 평균을 계산하여 문장 벡터 생성
    word_vectors = []
    for token in tokens:
        try:
            vector = word2vec_model.wv[token]
            word_vectors.append(vector)
        except KeyError:
            # 단어가 임베딩에 없는 경우 무시
            pass
    if word_vectors:
        sentence_vector = np.mean(word_vectors, axis=0)
    else:
        sentence_vector = np.zeros(word2vec_model.vector_size)
    return sentence_vector

def get_best_sentence(words):
    # 입력된 단어들을 토큰화하고 형태소 분석 수행
    tokens = preprocess_sentence(' '.join(words))
    
    # 입력된 단어들과 문장들 간의 코사인 유사도 측정
    max_similarity = -1
    best_sentence = ""
    for sentence in sentences:
        sentence_tokens = preprocess_sentence(sentence)
        sentence_vector = get_sentence_vector(sentence_tokens)
        input_vector = get_sentence_vector(tokens)
        similarity = cosine_similarity([input_vector], [sentence_vector])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            best_sentence = sentence
    return best_sentence

# 문장들
sentences = [
    "늘은 날씨가 정말 좋아서 밖에 나가야겠어.",
    "새로운 레스토랑을 발견해서 기분이 좋아.",
    "이번 주말에는 가족들과 함께 저녁을 먹을 거야.",
    "요즘 드라마를 보면서 시간을 보내고 있어.",
    "저는 오늘 너무 피곤해.",
    "그는 어제 새로운 차를 샀어.",
    "저는 주말에 항상 친구들을 만나요.",
    "그녀는 노래를 들으면서 운동을 해.",
    "지금은 밥 먹는 시간이야.",
    "이번 주에는 새로운 카페를 발견했어.",
    "학교에서는 친구들과 함께 공부했어.",
    "그는 요즘 공부에 열중하고 있어.",
    "이번 주말에는 새로운 장소로 여행을 가보고 싶어.",
    "우리는 함께 영화를 보러 가기로 했어.",
    "집에 가면 바로 저녁을 해야겠어.",
    "그녀는 어제 파티에 가서 즐거운 시간을 보냈어.",
    "오늘은 집에서 쉬는 날이야.",
    "친구들과 함께 점심을 먹었어.",
    "저는 노래를 들으면서 집안일을 해.",
    "오늘은 외출하기 좋은 날씨야.",
    "그는 친구들과 함께 축구를 하러 갔어.",
    "주말에는 항상 가족들과 함께 시간을 보내요.",
    "저는 운동을 할 때 음악을 들어요.",
    "그녀는 요즘 새로운 취미를 찾고 있어.",
    "이번 주에는 새로운 카페를 발견했어.",
    "집에 가면 바로 저녁을 해야겠어.",
    "그녀는 어제 파티에 가서 즐거운 시간을 보냈어.",
    "오늘은 집에서 쉬는 날이야.",
    "친구들과 함께 점심을 먹었어.",
    "저는 노래를 들으면서 집안일을 해.",
    "오늘은 외출하기 좋은 날씨야.",
    "그는 친구들과 함께 축구를 하러 갔어.",
    "주말에는 항상 가족들과 함께 시간을 보내요.",
    "저는 운동을 할 때 음악을 들어요.",
    "그녀는 요즘 새로운 취미를 찾고 있어."
]

# 테스트를 위한 단어 입력 받기
words = input("단어들을 입력하세요 (띄어쓰기로 구분): ").split()

# 가장 유사한 문장 출력
best_sentence = get_best_sentence(words)
print("가장 유사한 문장:", best_sentence)
