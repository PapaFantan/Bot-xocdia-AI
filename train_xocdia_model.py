import random
from sklearn.ensemble import RandomForestClassifier
import joblib

# Tạo dữ liệu giả lập
X = []
y = []

for _ in range(5000):
    seq = [random.choice([0, 1, 2, 3, 4]) for _ in range(3)]
    next_result = random.choice([0, 1, 2, 3, 4])
    X.append(seq)
    y.append(next_result)

# Huấn luyện mô hình
model = RandomForestClassifier()
model.fit(X, y)

# Lưu mô hình
joblib.dump(model, "xocdia_model.pkl")
print("✅ Đã huấn luyện và lưu mô hình AI!")