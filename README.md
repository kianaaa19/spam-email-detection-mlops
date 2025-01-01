# Submission Proyek Akhir: Spam Email Classification
Nama: Santanam Wishal

Username dicoding: kianaa19

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Spam email classification](https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification) |
| Masalah | Spam merupakan salah satu hal yang menganggu kenyamanan user dalam menggunakan email. |
| Solusi machine learning | Machine Learning Natural Language Processing dapat mendeteksi Spam pada email. |
| Metode pengolahan | Metode pengolahan data yang digunakan pada proyek ini berupa tokenisasi fitur input (text dari email) yang awalnya berupa text diubah menjadi susunan angka yang merepresentasikan text tersebut agar dapat dengan mudah dimengerti oleh model (Text Vectorization)|
| Arsitektur model | Model yang dibangun menggunakan layer TextVectorization sebagai layer yang akan memproses input string kedalam bentuk susunan angka, kemudian layer Embedding yang bertugas untuk mempelajari kedekatan atau kemiripan dari sebuah kata yang berguna untuk mengetahui apakah kata tersebut merupakan kata negatif atau kata positif. Lalu terdapat 2 hidden layer dan 1 output layer |
| Metrik evaluasi | Metrik yang digunakan pada model yaitu Binary Accuracy, True Positive, False Positive, True Negative, False Negative untuk mengevaluasi performa model dalam menentukan klasifikasi |
| Performa model | Model yang dibuat menghasilkan performa yang cukup baik dalam memberikan prediksi untuk text berita yang diinputkan, dan dari pelatihan yang dilakukan model menghasilkan binary_accuracy lebih dari 98% |
| Opsi deployment | Model dideploy di Cloudeka dengan menggunakan layanan DekaFlexi |
| Web app | [spam_ml](https://kianaa19-spam-detection-ml.w3spaces.com)  |
| Monitoring | Monitoring pada sistem ini dilakukan menggunakan prometheus dan grafana. Disini hanya dilakukan proses monitoring untuk menampilkan request yang masuk pada sistem yang akan menamplkan status pada tiap request yang dilakukan, pada sistem ini terdapat tiga status yang ditampilkan yaitu apabila proses request pada sistem klasifikasi not found, invalid argument dan proses klasifikasi berhasil ditandakan dengan ok |
