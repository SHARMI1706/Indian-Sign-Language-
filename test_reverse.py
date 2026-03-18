import joblib
import os
import cv2
import matplotlib.pyplot as plt

model = joblib.load("isl_number_reverse_model.pkl")

num = input("Enter a number (0-9): ").strip()

img_path = model.get(num)

if img_path and os.path.exists(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.title(f"ISL Number {num}")
    plt.axis("off")
    plt.show()
else:
    print("❌ Invalid input")
