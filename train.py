import pandas as pd  
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import train_test_split  
from sklearn.externals import joblib  
  
# 假设你已经训练好了一个模型，命名为 model  
# joblib.dump(model, 'model.pkl')
  
# 加载训练数据  
data = pd.read_csv('politics_questions.csv')  
  
# 将问题和答案分开  
questions = data['问题']  
answers = data['答案']  
  
# 将问题转换为向量形式  
vectorizer = CountVectorizer()  
question_vectors = vectorizer.fit_transform(questions)  
  
# 将答案转换为数字形式（0 或 1）  
answer_numbers = answers.map({'错误': 0, '正确': 1})  
  
# 将数据集分为训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(question_vectors, answer_numbers, test_size=0.2)  
  
# 创建逻辑回归模型并训练  
model = LogisticRegression()  
model.fit(X_train, y_train)  

# 保存！
joblib.dump(model, 'model.pkl')


# 测试模型在测试集上的准确率  
accuracy = model.score(X_test, y_test)  
print('准确率:', str(accuracy * 100) + '%')  
  
# 使用模型进行预测  
if __name__ == "__main__":
    example_question = '人民代表大会制度是我国的根本政治制度。'  
    example_vector = vectorizer.transform([example_question])  
    predicted_answer = model.predict(example_vector)  
    print('预测答案:', '正确' if predicted_answer[0] else '错误')