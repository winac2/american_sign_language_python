'''
#    #   ###   #   #  #   #  #####  #    #      ##   #     ###   #   #  #   #    #  #  #   #   ###   #
##   #  #   #  #   #   # #   #      ##   #     #  #        #  #  #   #   # #     # #   #   #  #   #
# #  #  #  ##  #   #    #    #####  # #  #      #    #     #  #  #   #    #      ##    #   #  #   #  #
#  # #  #      #   #    #    #      #  # #       #   #     #  #  #   #    #      ##    #####  #   #  #
#   ##  #   #  #   #    #    #      #   ##     #  #  #     #  #  #   #    #      # #   #   #  #   #  #
#    #   ###    ###     #    #####  #    #      ##   #     ###    ###     #      #  #  #   #   ###   #

'''
'''by "winac2-NSDK" - ndkhoi234@gmail.com'''
import os
import numpy as np
import cv2
import mediapipe as mp

# Đường dẫn dataset và file lưu dữ liệu sau xử lý
DATASET_DIR = './dataset'
OUTPUT_CSV_PATH = './data/processed_hand_data.csv'

# Khởi tạo MediaPipe 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# tạo danh sách dữ liệu
data = []

# Duyệt qua từng thư mục trong dataset
classes = os.listdir(DATASET_DIR)
print(f"Classes detected: {classes}")

for class_name in classes:
    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    print(f"Processing class: {class_name}")
    image_files = os.listdir(class_dir)

    for image_file in image_files:
        image_path = os.path.join(class_dir, image_file)
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Error reading {image_path}, skipping.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Lấy các tọa độ (x, y) của landmarks
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]

                # Tính vector đặc trưng ( để chuẩn hóa tọa độ)
                feature_vector = []
                for i in range(len(hand_landmarks.landmark)):
                    feature_vector.append(x_coords[i] - min(x_coords))
                    feature_vector.append(y_coords[i] - min(y_coords))

                # Gắn nhãn vào dữ liệu
                feature_vector.append(class_name)  # Class label
                data.append(feature_vector)

        else:
            print(f"No hand landmarks detected in {image_path}, skipping.")

# Chuyển dữ liệu thành mảng numpy
data = np.array(data, dtype=object)

# Lưu dữ liệu vào file CSV
if data.size > 0:
    np.savetxt(OUTPUT_CSV_PATH, data, delimiter=",", fmt='%s')
    print(f"Processed data saved to {OUTPUT_CSV_PATH}")
else:
    print("No data processed, nothing to save.")

# Kết thúc
hands.close()
print("Processing completed.")
