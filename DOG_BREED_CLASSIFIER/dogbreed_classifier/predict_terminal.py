import os
import cv2
import numpy as np
import pickle
import hashlib
import re

# Model dosyalarını yükleme
model_path = "classifier/utils/"
with open(os.path.join(model_path, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(model_path, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(model_path, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

# Görsel hash hesaplama fonksiyonu
def calculate_image_hash(image_path):
    """Bir görselin hash değerini hesaplar."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Görseli gri tonlamaya çevir
    img_resized = cv2.resize(img, (64, 64))  # Görseli sabit boyuta küçült
    return hashlib.md5(img_resized).hexdigest()  # Görselin hash değerini döndür

def natural_sort_key(string):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', string)]

# Tahmin fonksiyonu
def predict_image(image_path):
    """Bir görselin tahmin edilen sınıfını döndürür."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Görsel okunamadı! Lütfen desteklenen bir dosya sağlayın.")
    img_resized = cv2.resize(img, (64, 64)).flatten().reshape(1, -1)
    img_scaled = scaler.transform(img_resized)
    prediction = model.predict(img_scaled)
    return label_encoder.inverse_transform(prediction)[0]

# Gerçek sınıfı bulma fonksiyonu
def find_real_class(image_path, dataset_path):
    """Bir görselin dataset içinde gerçek sınıfını bulur."""
    uploaded_image_hash = calculate_image_hash(image_path)
    for class_name in os.listdir(dataset_path):  # Dataset'teki her sınıfı kontrol et
        class_folder = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_folder):
            for file_name in os.listdir(class_folder):  # Her görseli kontrol et
                file_path = os.path.join(class_folder, file_name)
                if os.path.isfile(file_path):
                    dataset_image_hash = calculate_image_hash(file_path)
                    if uploaded_image_hash == dataset_image_hash:  # Hash'ler eşleşiyor mu?
                        return class_name
    return "Seçilen veri dataset dışından olduğu için belirlenemedi"

# Test klasörü ve dataset yolu
test_folder_path = r"C:\Users\90545\Desktop\Test_Images"
dataset_path = r"C:\Users\90545\Desktop\DogBreedClassifier\dataset"

def main():
    if not os.path.exists(test_folder_path):
        print(f"Hata: {test_folder_path} klasörü bulunamadı.")
        return

    # Doğal sıralama
    for image_name in sorted(os.listdir(test_folder_path), key=natural_sort_key):
        image_path = os.path.join(test_folder_path, image_name)

        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Tahmin edilen sınıfı bul
                predicted_class = predict_image(image_path)

                # Gerçek sınıfı bul
                real_class = find_real_class(image_path, dataset_path)

                # Sonuçları yazdır
                print(f"Dosya: {image_name}")
                print(f"  Tahmin edilen sınıf: {predicted_class}")
                print(f"  Gerçek sınıf: {real_class}")
                print("-" * 50)

            except Exception as e:
                print(f"Dosya: {image_name}")
                print(f"  Bir hata oluştu: {str(e)}")
                print("-" * 50)

if __name__ == "__main__":
    main()