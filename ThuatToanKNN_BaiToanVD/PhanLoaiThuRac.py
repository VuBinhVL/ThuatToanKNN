# Bước 1: Import các thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import fetch_openml

# Bước 2: Load dữ liệu SpamBase từ openml
data = fetch_openml(name="spambase", version=1)
df = pd.DataFrame(data=data['data'], columns=data['feature_names'])
df['target'] = data['target']

# Bước 3: Tách tập dữ liệu thành features và labels
X = df.drop('target', axis=1)  # Dữ liệu đặc trưng
y = df['target'].astype(int)    # Nhãn (Spam = 1, Not spam = 0)

# Bước 4: Chia dữ liệu thành tập huấn luyện và tập kiểm tra (Train-test split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Bước 5: Khởi tạo mô hình KNN với k = 5
knn = KNeighborsClassifier(n_neighbors=5)

# Bước 6: Huấn luyện mô hình trên tập huấn luyện
knn.fit(X_train, y_train)

# Bước 7: Dự đoán trên tập kiểm tra
y_pred = knn.predict(X_test)

# Bước 8: Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Độ chính xác của mô hình: {accuracy*100:.2f}%")
print("Ma trận nhầm lẫn:")
print(conf_matrix)
print("\nBáo cáo phân loại:")
print(class_report)
