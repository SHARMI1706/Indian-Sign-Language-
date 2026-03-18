import os
import joblib

BASE_DIR = "numbers_visual"

reverse_model = {}

print("Checking files in:", BASE_DIR)
print("Files found:", os.listdir(BASE_DIR))

for i in range(10):
    filename = f"{i}.png"
    path = os.path.join(BASE_DIR, filename)
    print("Checking:", path)

    if os.path.exists(path):
        reverse_model[str(i)] = path

joblib.dump(reverse_model, "isl_number_reverse_model.pkl")

print("✅ Model created:", reverse_model)
