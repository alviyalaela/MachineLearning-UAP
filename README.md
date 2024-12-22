
# Klasifikasi Penyakit pada Buah Tomat dengan Deteksi Daun Menggunakan Algoritma MobileNetV2 dan ResNet101

## Project Overview

Project ini bertujuan untuk membuat sebuah model machine learning yang dapat mengklasifikasikan beberapa jenis penyakit pada buah tomat. Tomat (Solanum lycopersicum) merupakan salah satu tanaman hortikultura yang memiliki peran penting dalam sektor pertanian dan ekonomi. Produksi tomat yang optimal sangat dipengaruhi oleh kesehatan tanaman, di mana keberadaan penyakit daun pada tanaman tomat dapat mengurangi hasil panen secara signifikan. Beberapa penyakit umum yang menyerang daun tomat, seperti Early Blight, Late Blight, Leaf Mold, Mosaic Virus, Septoria Spot, Bacterial Spot, dan Yellow Leaf Curl Virus.

## Dataset Used

Dataset yang digunakan dalam project ini berasal dari [Mendeley Data](https://data.mendeley.com/datasets/b6cj8k8x3g/3). Dalam dataset ini terdapat beberapa citra daun tanaman, namun author **hanya mengambil dataset tomat.**

## Dataset Overview
Terdapat total 8 kelas pada dataset dengan 745 di setiap kelasnya. Total citra adalah 5960.

- bacterial_spot: 745 images
- early_blight: 745 images
- healthy: 745 images
- late_blight: 745 images
- leaf_mold: 745 images
- mosaic_virus: 745 images
- septoria_spot: 745 images
- yellow_leaf_curl: 745 images

Berikut adalah contoh masing-masing citra setiap kelas:

![image](https://github.com/alviyalaela/MachineLearning-UAP/blob/main/Assets/Image_samples.png?raw=true)

## Image Preprocessing
Setiap citra dalam dataset diresize dengan ukuran (224, 224), kemudian diaugmentasi dengan parameter sebagai berikut:
![image](https://github.com/alviyalaela/MachineLearning-UAP/blob/main/Assets/Augmentation_Parameter.png?raw=true)

Berikut adalah contoh dataset yang telah diaugmentasi:
![image](https://github.com/alviyalaela/MachineLearning-UAP/blob/main/Assets/Augmented_images.png?raw=true)

Setelah dilakukan *resize* dan *augmentasi,* citra displit menjadi data *training, validation,* dan *testing* dengan proporsi 80:10:10. Berikut hasil *splitting:*
![image](https://github.com/alviyalaela/MachineLearning-UAP/blob/main/Assets/Splitting_Data.png?raw=true)


## Algorithm Used

Dalam project ini, author menggunakan algoritma [MobileNetV2](https://keras.io/api/applications/mobilenet/) dan [ResNet101](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html#resnet101). Kedua algoritma dibandingkan untuk menghasilkan model terbaik.

**MobileNetV2 Architecture**
![image](https://www.researchgate.net/publication/350152088/figure/fig1/AS:1002717703045121@1616077938892/The-proposed-MobileNetV2-network-architecture.png)


**ResNet Architecture**
![image](https://www.researchgate.net/publication/378738074/figure/fig2/AS:11431281238761509@1714058388876/Structure-of-the-ResNet101-model.tif)

## Model Output
Berikut adalah hasil dari model yang telah dibangun:
- MobileNetV2
![image](https://github.com/alviyalaela/MachineLearning-UAP/blob/main/Assets/MobileNet_Learning%20Curve.png?raw=true)
Akurasi training meningkat pesat hingga sekitar 5 epoch, kemudian mendekati stabil dan hampir mencapai nilai 1 (atau 100%). Akurasi validasi cenderung stabil setelah awal pelatihan dan mencapai nilai tinggi yang serupa dengan akurasi pelatihan. Akurasi validasi tinggi yang hampir setara dengan akurasi pelatihan menunjukkan **model tidak mengalami overfitting.**

- ResNet101
![image](https://github.com/alviyalaela/MachineLearning-UAP/blob/main/Assets/ResNet_Learning%20Curve.png?raw=true)
Training loss terus menurun selama training. Hal ini normal karena model belajar memprediksi data training dengan lebih baik.
Validation loss juga terus menurun, yang menunjukkan bahwa **model tidak mengalami overfitting.**


## Installation
**Instal dependencies:**
   Instal library yang dibutuhkan dengan menjalankan perintah berikut:
   ```bash
   pip install matplotlib opencv-python numpy pandas seaborn rembg scikit-image scikit-learn tensorflow
   ```
   Tambahan library:
   - Mount Google Drive (khusus untuk Google Colab):
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Library lainnya yang sudah otomatis terpasang dalam lingkungan Google Colab, seperti `os`, `random`.

**Jalankan aplikasi:**
   ```bash
   python main.py
   ```

**Akses aplikasi web** melalui browser di alamat: `https://tomato-disease-classifier.streamlit.app/`.

## Author 👨‍💻 
- [@Alviya Laela](https://github.com/alviyalaela)
