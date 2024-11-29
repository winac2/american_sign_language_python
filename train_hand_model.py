'''
#    #   ###   #   #  #   #  #####  #    #      ##   #     ###   #   #  #   #    #  #  #   #   ###   #
##   #  #   #  #   #   # #   #      ##   #     #  #        #  #  #   #   # #     # #   #   #  #   #
# #  #  #  ##  #   #    #    #####  # #  #      #    #     #  #  #   #    #      ##    #   #  #   #  #
#  # #  #      #   #    #    #      #  # #       #   #     #  #  #   #    #      ##    #####  #   #  #
#   ##  #   #  #   #    #    #      #   ##     #  #  #     #  #  #   #    #      # #   #   #  #   #  #
#    #   ###    ###     #    #####  #    #      ##   #     ###    ###     #      #  #  #   #   ###   #

'''
'''by "winac2-NSDK" - ndkhoi234@gmail.com'''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

#dẫn đến file dữ liệu
processed_data_path = './data/processed_hand_data.csv'  #lấy data
model_save_path = './models/hand_gesture_model.pkl'    #nơi lưu model ( file models)

# Đọc data từ file CSV
data = np.genfromtxt(processed_data_path, delimiter=',', dtype=object)

# Tách đặc trưng (X) và nhãn (y)
X = np.array(data[:, :-1], dtype=float)  # Các cột trừ cột cuối là đặc trưng
y = np.array(data[:, -1], dtype=str)    # Cột cuối là label

# Chia tập dữ liệu thành train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Khởi tạo và train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Đánh giá model và thông báo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Lưu model vào file
with open(model_save_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to {model_save_path}")
