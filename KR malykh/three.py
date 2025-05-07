from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

#Структура данных
print("Количество изображений:", digits.data.shape[0])
print("Размер одного изображения:", digits.data.shape[1], "пикселей (8x8)")
print("Классы (цифры от 0 до 9):", digits.target_names)

#Вывод 10 изображений
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"Метка: {digits.target[i]}")
    ax.axis('off')
plt.show()

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

#Обучение SVM
model = SVC()
model.fit(X_train, y_train)

#Предсказание и точность
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.2f}")

#Ошибки
errors = (y_pred != y_test)
error_indices = [i for i, err in enumerate(errors) if err]

#Первые 5 ошибок
plt.figure(figsize=(12, 3))
for i, idx in enumerate(error_indices[:5]):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
    plt.title(f"Предсказано: {y_pred[idx]}\nИстина: {y_test[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel('Предсказанные метки')
plt.ylabel('Истинные метки')
plt.title('Матрица ошибок')
plt.show()