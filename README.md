# Face_Net
Fungsi praktikum ini adalah membuat sistem verifikasi dan identifikasi wajah. Kamu belajar cara deteksi wajah, alignment, ekstraksi embedding 512 dimensi, lalu membandingkan kemiripan atau mengklasifikasikan identitas.
```
FaceNet/
│
├── data/
│   ├── train/
│   │   ├── aditya/
│   │   │   ├── Aditya_1.jpg #atur sesuai folder anda
│   │   │   └── Aditya_2.jpg #atur sesuai folder anda
│   │   └── rusdi/
│   │       ├── Rusdi_1.jpg #atur sesuai folder anda
│   │       └── Rusdi_2.jpg #atur sesuai folder anda
│   │
│   └── val/
│       ├── aditya/
│       │   ├── Aditya_1.jpg #atur sesuai folder anda
│       │   └── Aditya_2.jpg #atur sesuai folder anda
│       └── rusdi/
│          ├── Rusdi_1.jpg #atur sesuai folder anda
│          └── Rusdi_2.jpg #atur sesuai folder anda
│         
├── build_embeddings.py
├── eval_folder.py
├── facenet_svm.joblib
├── predict_one.py
├── train_classifier.py
├── train_knn.py
├── utils_facenet.py
├── verify_cli.py
├── verify_pair.py
```

# Tujuan Praktikum
1.Memahami deteksi dan alignment wajah dengan MTCNN
2.Menghasilkan embedding FaceNet dimensi 512
3.Melakukan verifikasi wajah satu lawan satu
4.Melakukan identifikasi wajah menggunakan SVM
5.Mengevaluasi akurasi model pada data validasi

# Analisis Setiap Kode File
## utils_facenet.py
Berdasarkan analisis mendalam terhadap kode utils_facenet.py, dapat disimpulkan bahwa berkas ini berperan sebagai komponen inti dalam proses ekstraksi fitur wajah menggunakan FaceNet. Implementasi kode ini menggabungkan beberapa tahapan penting, mulai dari pembacaan gambar menggunakan OpenCV, konversi format untuk kompatibilitas dengan MTCNN, hingga proses deteksi dan alignment wajah yang dilakukan secara otomatis. Pemanfaatan MTCNN dan InceptionResnetV1 pretrained VGGFace2 menunjukkan bahwa desain kode berorientasi pada akurasi tinggi dalam ekstraksi embedding 512 dimensi. Selain itu, pemisahan fungsi menjadi beberapa blok seperti read_img_bgr, face_align, embed_face_tensor, dan embed_from_path menjadikan struktur kode lebih modular dan mudah dirawat. Keunggulan utama file ini terletak pada pipeline yang sederhana namun efektif, mekanisme penanganan kesalahan yang memadai, serta fleksibilitas penggunaan baik untuk prediksi tunggal maupun integrasi ke model klasifikasi. Meskipun sudah optimal, kode ini tetap memiliki potensi pengembangan seperti penambahan validasi format gambar dan mekanisme caching untuk meningkatkan performa pemrosesan pada dataset besar.

## train_classifier.py
Berdasarkan analisis terhadap train_classifier.py, file ini merupakan implementasi model klasifikasi berbasis Logistic Regression yang didesain untuk mengenali identitas wajah berdasarkan embedding output FaceNet. Kode ini menampilkan pendekatan pipeline yang terstruktur rapi melalui integrasi StandardScaler dan LogisticRegression di dalam satu alur pemrosesan menggunakan Pipeline dari scikit-learn. Pemilihan Logistic Regression sebagai pengganti SVM menunjukkan orientasi terhadap kecepatan training, stabilitas pada dataset kecil, serta kemudahan tuning. Selain itu, proses memuat data embedding dari X_train.npy dan y_train.npy dilakukan secara langsung, yang menjadikan file ini sederhana namun efektif. Kode ini juga menyediakan evaluasi akurasi secara manual setelah training, sehingga pengguna dapat memperoleh gambaran performa model secara cepat. Keunggulannya terletak pada kesederhanaan, konsistensi pipeline, serta kompatibilitas dengan modul prediksi. Namun, file ini masih dapat ditingkatkan melalui penambahan cross-validation, normalisasi data lebih fleksibel, serta konfigurasi hyperparameter yang lebih adaptif terhadap variasi dataset.

## train_knn.py
Berdasarkan analisis mendalam terhadap train_knn.py, dapat dilihat bahwa file ini mengimplementasikan model K-Nearest Neighbors (KNN) sebagai alternatif klasifier berbasis jarak untuk embedding FaceNet. Penggunaan pipeline StandardScaler → KNN menjadikan alur training lebih terstruktur dan tetap konsisten dengan model lainnya. KNN dengan metric euclidean sangat cocok dipadukan dengan embedding 512 dimensi karena sifatnya yang mengandalkan jarak antar titik di ruang vektor. Kode ini menampilkan pendekatan yang ringkas namun efektif, terutama untuk dataset kecil hingga menengah. Keunggulan file ini adalah kesederhanaannya, minim risiko overfitting, serta sifat non-parametrik yang membuatnya cocok untuk kasus di mana jumlah data per kelas sedikit. Meskipun begitu, performa KNN dapat menurun pada dataset besar karena kompleksitas pencarian jarak, sehingga perlu pengembangan seperti KD-tree, parameter tuning, atau penerapan threshold tambahan untuk mendeteksi wajah yang tidak dikenal.

## predict_one.py
Berdasarkan analisis terhadap predict_one.py, dapat dipahami bahwa file ini berfungsi sebagai modul prediksi tunggal yang memanfaatkan model klasifikasi yang telah dilatih sebelumnya. Kode ini mengimplementasikan proses embed → prediksi → penentuan kelas secara efisien dalam satu fungsi bernama predict_image. File ini memuat model facenet_svm.joblib, menghitung probabilitas kelas, dan menerapkan mekanisme threshold untuk mengklasifikasikan wajah yang tidak dikenal sebagai UNKNOWN. Pendekatan ini memberikan fleksibilitas dan keamanan tambahan dalam sistem pengenalan wajah karena tidak memaksakan prediksi jika confidence terlalu rendah. Desain kode yang ringkas dan modular menjadikannya mudah digunakan baik di command line maupun integrasi ke aplikasi backend. Meskipun sudah cukup matang, peningkatan seperti logging, visualisasi confidence, atau dukungan batch prediction dapat membuat file ini lebih adaptif pada penggunaan skala besar.

## verify_cli.py
Analisis verify_cli.py menunjukkan bahwa file ini dirancang untuk melakukan verifikasi wajah 1:1 melalui command line interface. Kode ini menerima dua path gambar sebagai input, kemudian menghitung kemiripan embedding menggunakan cosine similarity. Desainnya sederhana dan fokus, memungkinkan pengguna melakukan perbandingan wajah secara langsung tanpa proses klasifikasi model. Threshold yang dapat disesuaikan memberikan fleksibilitas terhadap tingkat sensitivitas verifikasi. Keunggulannya adalah kecepatan eksekusi dan struktur kode yang minimalis, menjadikannya alat verifikasi yang ringan namun akurat. Meski demikian, pengembangan seperti menampilkan embedding, validasi ekstensi file, atau menambahkan output berbasis skor probabilistik dapat meningkatkan nilai praktis file ini.

## verify_pair.py
Berdasarkan analisis terhadap verify_pair.py, file ini merupakan versi lebih sederhana dari verify_cli.py yang digunakan untuk pengujian manual dua gambar tertentu. Kode ini fokus pada pembandingan wajah dari dua file yang path-nya ditulis langsung dalam script, menjadikannya cocok untuk eksperimen cepat selama development. Fungsi embed_from_path dan cosine_similarity digunakan untuk menentukan tingkat kesamaan wajah, sementara threshold 0.85 menjadi standar awal untuk menyatakan match. Keunggulannya adalah kesederhanaan dan efektivitasnya dalam melakukan uji coba internal. Namun, fleksibilitasnya terbatas karena path tidak dinamis sehingga file ini ideal hanya untuk debugging, bukan produksi. Peningkatan seperti integrasi argparse atau otomatisasi batch comparison dapat membuatnya lebih bermanfaat.

## build_embeddings.py
Berdasarkan analisis menyeluruh terhadap kode build_embeddings.py, dapat disimpulkan bahwa file ini merupakan komponen fundamental dalam pipeline ekstraksi embedding pada sistem FaceNet. Implementasinya mengadopsi pendekatan modular yang efektif melalui pemisahan fungsi iter_images dan build_matrix, dimana iter_images berperan sebagai generator yang menelusuri struktur direktori dataset secara otomatis berdasarkan nama folder yang sekaligus menjadi label kelas. Pendekatan ini tidak hanya membuat proses loading dataset menjadi lebih efisien, tetapi juga mengurangi penggunaan memori berkat sifat generator yang tidak memuat seluruh path sekaligus ke dalam RAM. Sementara itu, fungsi build_matrix mengatur alur utama ekstraksi embedding melalui pemanggilan embed_from_path untuk setiap gambar, disertai mekanisme pencatatan kesalahan yang kuat untuk menyaring gambar-gambar yang gagal terdeteksi wajahnya. Integrasi tqdm sebagai progress bar menjadi peningkatan signifikan dari sisi pengalaman pengguna, terutama saat memproses dataset yang besar. Secara keseluruhan, desain kode ini memfasilitasi workflow yang bersih, mudah diperluas, dan dapat dijalankan sebagai standalone script maupun bagian dari modul yang lebih besar, meskipun masih terdapat ruang pengembangan pada aspek validasi format file dan parameterisasi path untuk meningkatkan fleksibilitas dalam berbagai skenario penggunaan.

## eval_folder.py
File eval_folder.py merupakan modul evaluasi sederhana namun fungsional yang dirancang untuk mengukur performa model pengenalan wajah berbasis embedding secara langsung dari struktur folder validasi. Implementasi kode ini secara efektif memanfaatkan model klasifikasi yang telah dilatih, kemudian menilai akurasinya terhadap data pada folder data/val dengan pola subfolder per kelas. Proses prediksi dilakukan dengan melakukan embedding setiap gambar menggunakan embed_from_path, kemudian membandingkan prediksi dengan label sebenarnya berdasarkan nama folder. Desainnya juga mencakup pencatatan metrik per kelas melalui dictionary dinamis yang menghitung jumlah prediksi benar serta total sampel, sehingga memberikan gambaran granular mengenai performa model pada tiap identitas. Pendekatan ini sangat bermanfaat untuk mendeteksi kelas yang underperforming akibat kurangnya data atau kesalahan representasi embedding. Selain itu, kode ini tetap mempertahankan kesederhanaannya tanpa mengorbankan fungsi inti, menjadikannya alat evaluasi yang ringan namun esensial. Meski begitu, evaluasi dapat diperluas dengan menambahkan confusion matrix, ROC curve, atau threshold-based verification untuk menghasilkan analisis yang lebih komprehensif.
