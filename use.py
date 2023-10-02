from sklearn.externals import joblib  
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer  
  
NAME = '问题.txt'

vectorizer = CountVectorizer()
model = joblib.load('model.pkl')
questions = Path(NAME).read_text().rstrip().splitlines()

if __name__ == '__main__':
    for question in questions:
        question_vector = vectorizer.transform([question])
        predicted_answer = model.predict(question_vector)
        print('对于问题', question, '模型预测答案为', '正确' if predicted_answer[0] else '错误')