import os
import cv2
import numpy as np
import pickle
import hashlib
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings

# Modelleri global olarak yükleme
model_path = os.path.join(settings.BASE_DIR, "classifier/utils/")
with open(os.path.join(model_path, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(model_path, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(model_path, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

def calculate_image_hash(image_path):
    """Bir görselin hash değerini hesaplar."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Görseli gri tonlamaya çevir
    img_resized = cv2.resize(img, (64, 64))  # Görseli sabit boyuta küçült
    return hashlib.md5(img_resized).hexdigest()  # Görselin hash değerini döndür

def predict_image(request):
    context = {'image_url': None, 'predicted_class': None, 'real_class': None}

    if request.method == 'POST' and 'image' in request.FILES:
        # Fotoğrafı al ve kaydet
        image = request.FILES['image']
        img_path = default_storage.save(image.name, image)
        full_img_path = default_storage.path(img_path)  # Tam dosya yolunu al
        img_url = f"{settings.MEDIA_URL}{img_path}"  # Görselin URL'sini oluştur

        try:
            # Görseli işle
            img = cv2.imread(full_img_path)
            if img is None:
                raise ValueError("Görsel okunamadı! Lütfen desteklenen bir dosya yüklediğinizden emin olun.")

            img_resized = cv2.resize(img, (64, 64)).flatten().reshape(1, -1)
            img_scaled = scaler.transform(img_resized)
            prediction = model.predict(img_scaled)
            predicted_class = label_encoder.inverse_transform(prediction)[0]

            # Dataset içinde gerçek sınıfı kontrol et
            dataset_path = os.path.join(settings.BASE_DIR, 'dataset')
            real_class = None
            uploaded_image_hash = calculate_image_hash(full_img_path)  # Yüklenen görselin hash'ini hesapla

            for class_name in os.listdir(dataset_path):  # Her sınıf klasörünü gez
                class_folder = os.path.join(dataset_path, class_name)
                if os.path.isdir(class_folder):
                    for file_name in os.listdir(class_folder):  # Her dosyayı kontrol et
                        file_path = os.path.join(class_folder, file_name)
                        if os.path.isfile(file_path):
                            dataset_image_hash = calculate_image_hash(file_path)  # Dataset görselinin hash'ini hesapla
                            if uploaded_image_hash == dataset_image_hash:  # Hash'ler eşleşiyor mu?
                                real_class = class_name
                                break
                    if real_class:
                        break

            # Sonuçları bağlama ekle
            context['image_url'] = img_url  # Yüklenen görselin URL'si
            context['predicted_class'] = predicted_class  # Tahmin edilen sınıf
            context['real_class'] = real_class  # Dataset'teki gerçek sınıf (varsa)

        except Exception as e:
            context['error'] = f"Bir hata oluştu: {str(e)}"

    return render(request, 'predict.html', context)
