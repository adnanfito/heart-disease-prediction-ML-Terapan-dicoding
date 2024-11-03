# Laporan Proyek Machine Learning - Adnan Fito Dharmawan

## Prediksi Penyakit Jantung

Penyakit jantung merupakan salah satu penyebab utama kematian di seluruh dunia, termasuk di Indonesia. Menurut data dari Organisasi Kesehatan Dunia (WHO), sekitar 17,9 juta orang meninggal setiap tahunnya karena penyakit kardiovaskular, yang mencakup sekitar 31% dari total kematian global. Di Indonesia sendiri, data dari Kementerian Kesehatan menunjukkan bahwa penyakit jantung koroner menempati urutan pertama dalam penyebab kematian pada usia dewasa. Penyebab umum penyakit jantung meliputi tekanan darah tinggi, kadar kolesterol yang tinggi, kebiasaan merokok, serta faktor genetika.

Pengembangan model prediksi penyakit jantung menjadi sangat penting sebagai alat yang dapat membantu klinisi dan masyarakat umum untuk mendeteksi risiko penyakit ini sejak dini. Model ini dapat berperan penting dalam mendukung upaya preventif, membantu mengurangi angka kejadian penyakit jantung, serta menurunkan biaya perawatan kesehatan jangka panjang.

**Mengapa masalah ini penting untuk diselesaikan:**
* Penyakit jantung merupakan penyebab utama kematian di seluruh dunia, sehingga
deteksi dini dapat mengurangi angka mortalitas.
* Dengan mengetahui risiko secara dini, pasien dapat melakukan perubahan gaya hidup yang lebih sehat untuk mengurangi risiko penyakit.
* Model prediksi penyakit jantung dapat mendukung tenaga medis dalam mengambil keputusan klinis yang lebih informatif dan tepat sasaran.
* Penggunaan teknologi prediktif dalam kesehatan dapat mengurangi biaya perawatan jangka panjang karena masalah kesehatan dapat dicegah sebelum berkembang.

**Bagaimana masalah ini dapat diselesaikan:**
* Membangun model prediksi berbasis machine learning dengan memanfaatkan data kesehatan (misalnya usia, tekanan darah, kadar kolesterol, dan riwayat keluarga).
* Menggunakan algoritma seperti regresi logistik, pohon keputusan, atau jaringan saraf tiruan untuk mengidentifikasi pola risiko yang terkait dengan penyakit jantung.
* Mengimplementasikan model prediksi ini sebagai alat bantu di klinik atau aplikasi kesehatan, sehingga masyarakat umum bisa memantau risiko kesehatan jantung secara mandiri.
* Menyediakan model dengan antarmuka yang mudah digunakan agar tenaga kesehatan dan pengguna awam dapat dengan mudah memahami dan menggunakan hasil prediksi.

**Hasil Riset dan Referensi Terkait**
Beberapa penelitian yang telah dilakukan sebelumnya menunjukkan keberhasilan model machine learning dalam memprediksi risiko penyakit jantung. Beberapa referensi yang bisa menjadi acuan adalah sebagai berikut:

* Rajkumar, A., & Hariharan, M. (2020). Prediction of Heart Disease using Machine Learning Algorithms. International Journal of Research in Engineering and Technology.

* Chaurasia, V., & Pal, S. (2014). Data Mining Approach to Detect Heart Diseases. International Journal of Advanced Computer Science and Information Technology.

## Business Understanding

Penyakit jantung merupakan penyebab utama kematian di banyak negara, termasuk Indonesia. Meskipun upaya untuk mengedukasi masyarakat tentang gaya hidup sehat sudah dilakukan, angka kasus penyakit jantung masih tinggi. Mengingat keterbatasan tenaga medis dan waktu yang dibutuhkan untuk evaluasi klinis, solusi berbasis kecerdasan buatan diperlukan untuk membantu mengidentifikasi risiko penyakit jantung lebih cepat dan efisien. Dengan bantuan teknologi ini, baik pasien maupun tenaga kesehatan dapat membuat keputusan yang lebih tepat tentang tindakan preventif dan perawatan.

Bagian laporan ini mencakup:

### Problem Statements

- Bagaimana kita dapat mengembangkan model prediksi penyakit jantung yang akurat menggunakan data medis?
- Algoritma dan metode apa yang paling efektif dalam memberikan prediksi risiko penyakit jantung dengan akurasi tinggi?


### Goals

- Mengembangkan model machine learning yang mampu memprediksi risiko penyakit jantung berdasarkan faktor-faktor risiko seperti usia, tekanan darah, kadar kolesterol, dan lain-lain.
- Mencapai tingkat akurasi prediksi yang memadai untuk memberikan dukungan kepada tenaga medis dalam pengambilan keputusan klinis.

### Solution statements
Untuk mencapai tujuan yang telah ditentukan, beberapa solusi akan diterapkan, di antaranya:

**1. Menggunakan Beberapa Algoritma Machine Learning untuk Memilih Model Terbaik**
- Menggunakan dua algoritma untuk membangun model prediksi, seperti Random Forest dan Support Vector Machine. Masing-masing algoritma memiliki kelebihan dalam menangani data yang kompleks, dan hasilnya dapat dibandingkan untuk memilih model terbaik.
- Metrik Evaluasi: Metrik evaluasi yang digunakan adalah akurasi, precision, recall, dan F1-score untuk mengukur kinerja masing-masing model dalam memprediksi risiko penyakit jantung.

**2. Improvement pada Dataset dengan Metode Oversampling**
- Mengatasi masalah ketidakseimbangan kelas dalam data menggunakan metode oversampling dengan SMOTE *(Synthetic Minority Over-sampling Technique)* dan class_weight pada model. SMOTE menghasilkan sampel baru dari kelas minoritas, sedangkan class_weight membantu model untuk lebih mempertimbangkan kelas yang kurang dominan. Kedua metode ini bertujuan meningkatkan kemampuan model dalam memprediksi kelas hujan yang langka.
- Metrik evaluasi: Akurasi, Precision, Recall, dan F1-Score akan digunakan untuk mengevaluasi perbaikan kinerja model setelah penyeimbangan kelas.

## Data Understanding
Dataset yang digunakan berasal dari kaggle bernama [Heart Attack](https://www.kaggle.com/datasets/bharath011/heart-disease-classification-dataset) mendiagnosa pasien terkena penyakit jantung atau tidak.

### Jumlah Data dan Fitur
Dataset yang digunakan terdiri dari **1.319 entri** (data) dan **9 fitur**. Ini menunjukkan bahwa kita memiliki sejumlah data yang cukup untuk melakukan analisis dan pelatihan model machine learning.

### Ringkasan Data
Berdasarkan analisis awal, berikut adalah ringkasan informasi tentang dataset:

- **Total Entri**: 1.319
- **Fitur**: Terdiri dari 9 kolom, yang mencakup variabel input dan output.

| Nama Fitur      | Tipe Data | Deskripsi                                                                                       |
|------------------|-----------|-------------------------------------------------------------------------------------------------|
| `age`            | int64     | Usia individu dalam tahun.                                                                     |
| `gender`         | int64     | Kategori gender (0 untuk Perempuan, 1 untuk Laki-laki).                                        |
| `impluse`        | int64     | Detak jantung individu dalam denyut per menit (bpm).                                          |
| `pressurehight`  | int64     | Tekanan darah sistolik (mmHg) saat jantung memompa darah (nilai tekanan darah tertinggi).      |
| `pressurelow`    | int64     | Tekanan darah diastolik (mmHg) saat jantung beristirahat di antara denyutan (nilai terendah). |
| `glucose`        | float64   | Tingkat gula darah individu dalam mg/dL.                                                       |
| `kcm`            | float64   | Level Creatine Kinase-MB dalam ng/mL, enzim jantung sebagai indikator kerusakan otot jantung. |
| `troponin`       | float64   | Level Troponin dalam ng/mL, protein jantung yang meningkat saat ada kerusakan pada otot jantung. |
| `class`          | object    | Kategori hasil diagnosis: Negative (tidak ada serangan jantung, kelas 0) dan Positive (ada serangan jantung, kelas 1). |

### Kondisi Data
Berdasarkan analisis, tidak ditemukan nilai hilang pada setiap kolom dalam dataset. Berikut adalah jumlah nilai hilang per kolom:

| Fitur            | Jumlah Nilai Hilang |
|------------------|---------------------|
| `age`            | 0                   |
| `gender`         | 0                   |
| `impluse`        | 0                   |
| `pressurehight`  | 0                   |
| `pressurelow`    | 0                   |
| `glucose`        | 0                   |
| `kcm`            | 0                   |
| `troponin`       | 0                   |
| `class`          | 0                   |

Ini menunjukkan bahwa dataset dalam kondisi baik dan siap untuk diproses lebih lanjut tanpa perlu melakukan penanganan nilai hilang.

### Memori yang Digunakan
Dataset ini menggunakan memori sekitar **92.9 KB**, yang tergolong kecil dan efisien untuk diproses dalam analisis lebih lanjut.



**Correlation Matrix Heatmap Plot**
Melihat korelasi antar variabel
![img](https://i.imgur.com/a4DIR6y.png)

**KESIMPULAN** : 
* Korelasi **paling rendah** terhadap kolom `class` ada pada kolom `pressurelow` dan `impluse`
* Korelaasi **paling kuat** terhadap kolom `class` ada pada kolom `age`, `kcm`, dan `troponin`

**Distribusi Class Target**
![img](https://i.imgur.com/OgdUjt1.png)
**KESIMPULAN**: Dari diagram tersebut kita tahu bahwa target yang di prediksi tidak seimbang jumlahnya.

## Data Preparation
Pada bagian ini, dijelaskan tahapan-tahapan persiapan data yang dilakukan untuk memastikan data siap digunakan dalam proses pemodelan. Berikut adalah tahapan lengkap dalam Data Preparation:
1. Encoding Label Target
- Proses: Mengubah kolom Target menjadi nilai numerik secara manual.
- Alasan: Algoritma machine learning tidak bisa bekerja dengan data non-numerik, sehingga encoding diperlukan untuk mengonversi data kategorikal ke dalam bentuk yang bisa dimengerti oleh model.
2. Data Cleaning
- Proses: Menangani nilai kosong (missing values) pada dataset dengan melakukan imputasi. Pada kolom numerik, nilai kosong akan diisi menggunakan median atau rata-rata, sedangkan untuk kolom kategorikal, digunakan nilai modus.
- Alasan: Mengisi nilai kosong penting untuk memastikan proses pemodelan tidak terganggu, karena beberapa algoritma tidak bisa bekerja dengan data yang memiliki nilai kosong. Dengan imputasi ini, data menjadi lebih lengkap dan representatif.

3. Outlier Handling dengan Winsorizer
- Proses: Mengatasi outlier pada kolom numerik dengan teknik Winsorizing. Nilai ekstrem pada batas atas dan bawah distribusi diubah menjadi nilai persentil tertentu (misalnya 1% dan 99%) untuk mengurangi pengaruh ekstrem pada model.
- Alasan: Outlier bisa mengurangi akurasi model, terutama pada algoritma yang sensitif terhadap nilai ekstrem. Dengan Winsorizing, distribusi data menjadi lebih stabil sehingga model dapat mempelajari data dengan lebih efektif.
4. Normalization menggunakan RobustScaler
- Proses: Menerapkan RobustScaler untuk menormalkan data numerik dengan mengurangi median dan membaginya dengan IQR (Interquartile Range). Teknik ini membantu menstabilkan distribusi data tanpa terpengaruh oleh outlier.
- Alasan: Normalisasi diperlukan agar fitur berada pada skala yang sama, terutama bagi algoritma yang sensitif terhadap perbedaan skala. RobustScaler dipilih karena lebih tahan terhadap outlier dibandingkan metode lainnya.
5. Split Data (80:20)
- Proses: Memisahkan data menjadi data latih (train) dan data uji (test) dengan perbandingan 80:20. Data latih digunakan untuk melatih model, sedangkan data uji digunakan untuk mengevaluasi performa model.
- Alasan: Split data diperlukan untuk memastikan model tidak hanya belajar dari data latih, tetapi juga mampu bekerja dengan baik pada data baru yang belum pernah dilihat sebelumnya. Pembagian ini juga membantu mengevaluasi generalisasi model pada data uji.
6. Improvement pada Data
Peningkatan akurasi model dilakukan dengan teknik penyeimbangan data menggunakan SMOTE :
- SMOTE (Synthetic Minority Over-sampling Technique): Digunakan untuk menangani ketidakseimbangan kelas dengan membuat sampel sintetis dari kelas minoritas.

Setiap tahapan di atas dilakukan secara berurutan untuk memastikan data siap digunakan dalam proses pemodelan.

## Modeling
Bagian ini menjelaskan proses definisi dan pelatihan dua model machine learning untuk memprediksi Penyakit Jantung, yaitu Random Forest dan Support Vector Machine (SVM). Berikut adalah penjelasan masing-masing model beserta parameter yang digunakan:

#### 1. Random Forest
- Cara Kerja: Random Forest adalah algoritma berbasis ensemble yang menggunakan kombinasi dari banyak pohon keputusan (decision trees). Setiap pohon dibangun secara acak dengan subset data dan subset fitur, sehingga model dapat menangani berbagai variasi data dengan lebih baik dan memiliki daya tahan yang tinggi terhadap overfitting. Model ini sangat efektif untuk kasus klasifikasi dengan dataset kompleks, karena mampu memanfaatkan kekuatan kolektif dari beberapa pohon untuk mencapai akurasi yang tinggi.
- Parameter yang Digunakan:
    - `n_estimators=100`: Menentukan jumlah pohon dalam hutan. Lebih banyak pohon dapat meningkatkan kinerja model, tetapi juga meningkatkan waktu komputasi.
    - `random_state=42`: Seed untuk menjaga konsistensi hasil eksperimen.

#### 2. Support Vector Machine (SVM)
- Cara Kerja: SVM adalah algoritma klasifikasi yang bekerja dengan mencari hyperplane terbaik yang memisahkan kelas dalam ruang fitur. Dengan menggunakan kernel linear, SVM bertujuan memaksimalkan margin (jarak) antara titik data dari dua kelas yang berbeda, sehingga meminimalkan kesalahan klasifikasi. SVM sangat cocok untuk data yang distribusinya memungkinkan batas keputusan linear, sehingga model dapat memisahkan kelas dengan jelas.

- Parameter yang Digunakan:
    - `kernel='linear'`: Kernel linear dipilih karena efektif untuk memisahkan data yang dapat dibedakan secara linear.
    - `C=1`: Parameter regulasi yang mengontrol keseimbangan antara memaksimalkan margin dan meminimalkan kesalahan klasifikasi. Nilai C yang tinggi dapat menyebabkan overfitting, sedangkan nilai yang terlalu rendah bisa mengabaikan pola data penting.

### Kelebihan dan Kekurangan Algoritma yang Digunakan
- **Random Forest**

    - Kelebihan:
        - Tahan terhadap overfitting karena menggabungkan prediksi dari banyak pohon.
        - Memberikan estimasi pentingnya fitur, yang dapat membantu dalam interpretasi hasil.
        - Cenderung memiliki kinerja yang kuat pada data yang kompleks dan non-linear.

    - Kekurangan:
        - Membutuhkan lebih banyak memori dan waktu komputasi, terutama jika jumlah pohon besar.
        - Bisa sulit diinterpretasi karena terdiri dari banyak pohon.
- **Support Vector Machine (SVM):**

    - Kelebihan:
        - Mampu menghasilkan hyperplane dengan margin maksimum yang efektif untuk data yang memiliki batasan kelas yang jelas.
        - Tahan terhadap overfitting terutama dengan parameter regulasi yang tepat.
    - Kekurangan:
        - Waktu pelatihan lebih lama pada dataset yang besar.
        - Sensitif terhadap pilihan parameter, dan membutuhkan tuning untuk hasil yang optimal.


## Evaluation
Pada bagian ini, kami menggunakan beberapa metrik evaluasi untuk menilai kinerja model prediksi penyakit jantung yang dibangun menggunakan algoritma Support Vector Machine (SVM) dan Random Forest. Evaluasi dilakukan berdasarkan metrik Accuracy, Precision, Recall, dan F1-Score, yang dipilih karena relevan dengan konteks klasifikasi biner dalam data cuaca yang tidak seimbang.

### Metrik Evaluasi yang Digunakan
1. **Accuracy**
Akurasi adalah proporsi prediksi yang benar dari seluruh prediksi. Akurasi dihitung dengan rumus:
`Accuracy = (True Positives + True Negatives) / Total Observations`

2. **Precision**
Precision mengukur akurasi prediksi positif dengan rumus:
`Precision = True Positives / (True Positives + False Positives)`

3. **Recall**
Recall mengukur kemampuan model dalam mendeteksi semua kejadian positif (hujan) dengan rumus:
`Recall = True Positives / (True Positives + False Negatives)`

4. **F1-Score**
F1-Score adalah rata-rata harmonik dari precision dan recall, yang memberikan keseimbangan di antara keduanya, khususnya berguna pada data yang tidak seimbang.
`F1-Score = 2 * (Precision * Recall) / (Precision + Recall)`

## Hasil Evaluasi dan Pembahasan Terhadap Problem Statement dan Goals Proyek
### Hasil Proyek Berdasarkan Metrik Evaluasi
Berdasarkan hasil evaluasi performa kedua model, yaitu Random Forest (RF) dan Support Vector Machine (SVM), berikut adalah ringkasan skor akurasi dan metrik evaluasi lainnya dari masing-masing model setelah dilakukan oversampling menggunakan SMOTE :

| Model          | Oversampling | Akurasi Training | Akurasi Uji | Precision (0) | Recall (0) | F1-score (0) | Precision (1) | Recall (1) | F1-score (1) |
|----------------|--------------|------------------|-------------|---------------|------------|--------------|---------------|------------|--------------|
| Random Forest  | SMOTE        | 1.0000          | 0.9773      | 0.98          | 0.96       | 0.97         | 0.98          | 0.99       | 0.98         |
| SVM            | SMOTE        | 1.0000          | 0.9318      | 0.90          | 0.93       | 0.91         | 0.96          | 0.93       | 0.94         |

**Penjelasan:**
- Precision (0) dan Recall (0) menunjukkan performa model untuk kelas "0".
- Precision (1) dan Recall (1) menunjukkan performa model untuk kelas "1".

### Pembahasan Goal Utama Berdasarkan Hasil Prediksi
Model Random Forest menunjukkan performa yang sangat baik dalam klasifikasi risiko penyakit jantung dengan akurasi yang tinggi pada data uji (97.73%). Di sisi lain, SVM memiliki akurasi sedikit lebih rendah (93.18%), tetapi tetap memberikan hasil prediksi yang kuat, terutama pada data yang bersifat linier. Berdasarkan performa ini, model Random Forest dipilih sebagai model final untuk mendukung analisis risiko penyakit jantung yang akurat. Berikut adalah pembahasan hasil model dalam konteks tujuan proyek dan relevansi untuk mendukung layanan kesehatan dan prediksi risiko medis.

1. **Bagaimana kita dapat mengembangkan model prediksi penyakit jantung yang akurat menggunakan data medis?**
Dengan performa tinggi pada precision (0.98) dan recall (0.99) pada kelas risiko penyakit jantung, model Random Forest mampu menangani klasifikasi data medis yang kompleks dan beragam. Model ini memiliki kelebihan dalam mengenali pola-pola penting pada data yang mungkin sulit diidentifikasi oleh model linear seperti SVM. Keakuratan prediksi ini menunjukkan bahwa dengan pemilihan fitur yang tepat dan pengaturan parameter yang optimal, data medis dapat diolah menjadi model prediktif yang andal, terutama untuk aplikasi pada klinik atau rumah sakit yang memerlukan prediksi risiko penyakit jantung secara cepat.

    Precision tinggi pada kelas tanpa penyakit jantung (0.98) juga mengurangi risiko prediksi positif palsu, menghindari ketakutan atau kekhawatiran yang tidak perlu pada pasien. Hal ini memungkinkan dokter untuk mengandalkan prediksi ini dalam memberi keputusan klinis, sambil menjaga keseimbangan yang baik antara keakuratan dan risiko overfitting.

2. **Algoritma dan metode apa yang paling efektif dalam memberikan prediksi risiko penyakit jantung dengan akurasi tinggi?**
    Berdasarkan hasil akurasi, model Random Forest memiliki keunggulan dalam akurasi dan f1-score yang lebih tinggi dibandingkan SVM. Ini menunjukkan bahwa model ensemble seperti Random Forest lebih efektif untuk data medis yang mungkin memiliki variasi atau hubungan yang tidak linier antarfitur, yang sulit dipetakan dengan model SVM linear. Keunggulan Random Forest pada recall dan f1-score dalam mengklasifikasikan risiko penyakit jantung mengindikasikan kemampuan model ini untuk meminimalkan kesalahan dalam mengidentifikasi individu berisiko tinggi, suatu hal yang sangat penting dalam aplikasi medis.

    Pada sisi lain, SVM, dengan akurasi 93.18% dan kinerja yang baik dalam identifikasi kategori negatif, tetap berguna dalam pengembangan model prediksi, terutama jika dataset lebih kecil dan memiliki distribusi yang lebih linier. Namun, pada kasus ini, Random Forest lebih diandalkan karena kemampuan adaptifnya dalam memetakan hubungan kompleks dalam data medis.
    
### Goal Proyek Secara Keseluruhan

Dengan ketepatan prediksi risiko penyakit jantung yang ditunjukkan oleh model Random Forest, proyek ini berhasil mencapai tujuannya untuk memberikan prediksi yang andal sebagai alat bantu dalam penilaian risiko kesehatan. Model ini memiliki keseimbangan performa yang baik pada kelas risiko dan non-risiko, yang sangat relevan untuk digunakan dalam diagnosis medis awal atau sistem skrining. Meski demikian, performa pada data uji masih dapat ditingkatkan dengan eksplorasi fitur tambahan atau dengan lebih banyak data medis untuk menangani variasi kondisi pasien.

Kesimpulannya, model Random Forest yang dioptimalkan ini telah berhasil memenuhi tujuan utama proyek, yakni memberikan prediksi risiko penyakit jantung dengan keakuratan yang tinggi dan relevansi klinis yang kuat. Pengembangan selanjutnya dapat difokuskan pada peningkatan interpretasi model untuk memastikan bahwa model ini dapat diintegrasikan ke dalam sistem kesehatan yang lebih besar dan mendukung keputusan medis lebih akurat pada kasus yang sulit didiagnosis.
