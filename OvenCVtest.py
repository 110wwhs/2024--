from faker import Faker

# 한국어 faker 객체 생성
fake = Faker('ko_KR')

# 500개의 무작위 한국어 문장 생성
for _ in range(500):
    print(fake.sentence())
