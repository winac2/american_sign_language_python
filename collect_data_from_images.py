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
import cv2
import mediapipe as mp
import csv

DATASET_DIR = "dataset"  #lấy dữ liệu từ "dataset"
DATA_DIR = "data"      #nơi lưu dữ liệu xuất
HAND_DATA_FILE = os.path.join(DATA_DIR, "hand_data.csv")   #lưu thành file csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

def save_data(landmarks, label):
    """Lưu dữ liệu landmark vào file CSV"""
    with open(HAND_DATA_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([label] + landmarks)

def process_image(image_path):
    """Xử lý ảnh, xuất landmarks."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
            return landmarks
    return None

def main():
    """Duyệt qua từng folder trong dataset và thu thập dữ liệu."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(HAND_DATA_FILE):
        with open(HAND_DATA_FILE, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["label"] + [f"x{i}, y{i}, z{i}" for i in range(21)])

    for label_folder in os.listdir(DATASET_DIR):
        label_path = os.path.join(DATASET_DIR, label_folder)
        if os.path.isdir(label_path):
            print(f"Processing label: {label_folder}")
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                landmarks = process_image(image_path)
                if landmarks:
                    save_data(landmarks, label_folder)
                    print(f"Saved data for {image_path}")
                else:
                    print(f"Warning: No hand detected in {image_path}")

if __name__ == "__main__":
    main()
