# BISINDO Live Sign-Recognition Android App

Aplikasi Android native (Kotlin) yang mengenali isyarat BISINDO secara real-time dari kamera live:
kamera → ekstraksi keypoint tangan on-device (MediaPipe Hands) → inference **dua model** ONNX
(CNNsa-BiLSTM sebagai model utama + MViTv2 sebagai pembanding) → hasil, latency, dan pemakaian
CPU/RAM kedua model ditampilkan berdampingan. Sepenuhnya offline, tidak ada permission INTERNET.

Repo ini **terpisah total** dari repo riset ML Python (`CNN-LSTM_BISINDO_Android`) demi tetap
modular — satu-satunya keterkaitan adalah 2 file model (`cnnsa_bilstm.onnx`, `mvit_v2.onnx` +
`.onnx.data`) yang di-copy manual ke `app/src/main/assets/`, tanpa dependency source code,
submodule, atau shared build config.

## Status saat ini

**Selesai dan tervalidasi di device fisik** (Poco M4 Pro). Semua fitur yang direncanakan sudah
jalan end-to-end: rekam gestur, dua model berjalan & dibandingkan pada rekaman yang sama, riwayat
sesi dengan playback, tabel evaluasi lengkap (10-jendela CNN, latency, CPU/RAM, ukuran model), dan
ekspor data keypoint mentah (JSON) untuk diagnosis numerik. Log lengkap tiap keputusan, bug, dan
perbaikan (termasuk beberapa yang butuh beberapa iterasi sebelum benar-benar solid) ada di
[`PROGRESS.md`](PROGRESS.md) — baca itu dulu kalau lanjut sesi baru, dokumen ini fokus ke
"bagaimana aplikasi ini bekerja sekarang", bukan kronologi.

**Perjalanan singkat menuju kondisi sekarang** (detail penuh di `PROGRESS.md`):
CNNsa-BiLSTM sempat berjalan hampir sempurna di validasi desktop (95.31% akurasi test set) tapi
gagal total saat diuji live oleh pengguna sungguhan di HP — root cause-nya BUKAN kualitas model,
melainkan dua bug di sisi Android: (1) label handedness ("Left"/"Right" MediaPipe) live ternyata
kebalikan dari distribusi data training, dan (2) bitmap sensor mentah tidak pernah dirotasi ke
orientasi tegak sebelum diberikan ke MediaPipe. Setelah kedua bug ini diperbaiki dan divalidasi
ulang (replay 5 sesi live sungguhan lewat model ONNX asli di desktop: 0/5 benar → 5/5 benar,
~100% confidence), pengujian di device nyata mengonfirmasi perbaikannya (4/5 lalu 3/3 benar pada
sesi lanjutan). Di atas perbaikan itu ditambahkan strategi **10 jendela geser** ("sliding window")
per sesi rekam — alih-alih menuntut pengguna menebak persis detik keberapa harus mulai bergerak,
CNNsa-BiLSTM dijalankan atas 10 posisi jendela 30-frame yang tersebar di seluruh rekaman 5 detik,
dan hasil confidence tertinggi yang dipakai. Satu fitur (ekspor GIF flipbook ke luar aplikasi)
diimplementasikan penuh, lolos verifikasi independen di Python, tapi tetap crash berulang di
device tanpa akses `adb logcat` untuk mendiagnosis lebih lanjut — atas keputusan eksplisit,
fitur itu **dihapus** (preview flipbook in-app yang tidak pernah memakai kode bermasalah itu tetap
ada, cuma tombol unduh/share-nya yang dicabut).

## Fitur Utama

- **Rekam & Bandingkan** (`CameraActivity`): preview kamera live dengan overlay skeleton/bounding
  box tangan real-time, tombol "Rekam (5 dtk)" menjalankan CNNsa-BiLSTM (10 jendela geser, hasil
  confidence tertinggi jadi jawaban akhir) dan MViTv2 (satu klip 16-frame) atas rekaman yang sama,
  menampilkan gloss + confidence + latency **kedua model** berdampingan.
- **Riwayat** (`HistoryActivity` → `HistoryPlaybackActivity`): daftar sesi tersimpan diberi nomor
  kronologis ("Sesi N: <gloss>"), tap sesi untuk playback flipbook (frame JPEG dengan overlay
  sudah dipanggang langsung ke tiap frame) dan mengunduh/membagikan data keypoint mentah sesi itu
  sebagai `sequence.json` (30×126 float, persis yang dikirim ke CNNsa-BiLSTM) lewat share sheet
  Android — dipakai untuk diagnosis numerik tanpa perlu adb.
- **Evaluasi** (`EvaluationActivity`): tabel perbandingan CNNsa-BiLSTM (breakdown confidence
  ke-10 jendela geser, yang menang ditandai "(dipilih)") vs MViTv2 per sesi, latency inferensi,
  jumlah frame mentah terdeteksi vs frame yang benar-benar dipakai model (menyingkap throughput
  device), penggunaan CPU% dan RAM (PSS, MB) tiap model, ukuran model, plus grafik latensi.
- **Overlay skeleton tangan live**: digambar terus-menerus selama preview (termasuk saat merekam),
  bukan cuma kotak panduan statis — pengguna bisa melihat langsung titik/rangka tangannya sendiri
  untuk membidik gestur lebih presisi.

## Context

Bagian riset ML (repo `CNN-LSTM_BISINDO_Android`) sudah selesai: PreProcessing → FeatureExtraction
→ Training (CNNsa-BiLSTM 95.31% test accuracy, MViTv2 99.22%) → Export ke ONNX, keduanya
tervalidasi. **Kedua model** masuk ke aplikasi ini: CNNsa-BiLSTM (ringan, keypoint MediaPipe
126-dim, ONNX 14.65MB) sebagai model utama untuk inferensi real-time, dan MViTv2 (raw video
`(1,3,16,224,224)`, ONNX ~143MB termasuk external-data file) sebagai pembanding
akurasi-vs-efisiensi langsung di dalam aplikasi — bukan cuma dokumentasi/pembanding di luar app
seperti rencana awal, ini adalah salah satu fitur inti (layar Rekam & Bandingkan, layar Evaluasi).

Batasan desain yang tetap dipegang sejak awal:
- **Tidak pakai Android Studio** — dikerjakan dari **VSCode + terminal**, JDK + Android SDK
  command-line tools + Gradle.
- **Kotlin**, Android native (MediaPipe Hands tidak didukung resmi di Unity).
- **Repo/folder terpisah total** dari repo riset ML Python.
- **Input real-time via kamera** (bukan upload video).
- Device testing: HP fisik, **Poco M4 Pro** (Android 11, API 30), via USB debugging.

## Toolchain (tanpa Android Studio)

- **JDK 17**
- **Gradle 8.13** (bootstrap wrapper sekali, lalu `.\gradlew.bat` seterusnya)
- **Android SDK "Command line tools only"**, via `sdkmanager`: `platform-tools` (adb) +
  `platforms;android-35` + `platforms;android-36` + `build-tools;35.0.0` + `build-tools;36.0.0`
- **AGP 8.13.0** + **Kotlin plugin 2.1.20** (butuh JDK 17)
- `compileSdk=36` (androidx.camera:*:1.6.1 mensyaratkan compileSdk ≥ 36 — ketauan dari
  `assembleDebug` gagal di `checkDebugAarMetadata` sampai dinaikkan dari draft awal 35),
  `targetSdk=34`, `minSdk=26` — `targetSdk`/`minSdk` tidak berubah, jadi kompatibilitas device
  (Poco M4 Pro Android 11/API 30) tidak terpengaruh.
- Testing: Poco M4 Pro via USB debugging + `adb install -r` / `adb logcat` untuk sesi debugging
  aktif (dipakai berulang kali selama iterasi fitur di atas).

### VSCode extensions

Build/run sepenuhnya lewat terminal (`.\gradlew.bat assembleDebug`, `adb install -r`); extension
di bawah ini murni untuk editing.

| Extension | ID | Fungsi |
|---|---|---|
| Kotlin | `fwcd.kotlin` | Syntax highlighting + language server untuk file `.kt`. |
| Gradle for Java | `vscjava.vscode-gradle` | Panel sidebar untuk jalankan Gradle task tanpa hafal command. |
| XML | `redhat.vscode-xml` | Syntax highlight + validasi untuk manifest & layout XML. |

## Preprocessing real-time & replikasi training (inti teknis aplikasi)

Training pipeline (Python) memproses video yang sudah direkam secara batch. Aplikasi live tidak
punya langkah preprocessing terpisah — logika yang sama dijalankan terus-menerus per frame:

1. Kamera menangkap frame terus-menerus; setiap frame yang berhasil dianalisis MediaPipe langsung
   diproses jadi vector 126 angka (lihat aturan di bawah). Overlay skeleton diupdate dari hasil
   yang sama, live, tidak menunggu sesi rekam selesai.
2. Selama sesi "Rekam" (durasi tetap 5 detik, **bukan** "berhenti begitu 30 frame terkumpul" —
   desain awal itu bikin sesi rekam sering gagal begitu ada kerja ekstra per-frame yang
   memperlambat throughput MediaPipe), setiap hasil deteksi yang lolos throttle ~15fps (menyamai
   `target_fps` training) disimpan bersama timestamp elapsed-nya.
3. Di akhir rekaman, `KeypointSequenceBuilder.alignToSlidingWindows()` memotong seluruh rekaman
   jadi **10 jendela 30-frame** dengan titik mulai tersebar merata (berbasis timestamp, bukan
   index array — device yang throughput-nya jauh dari 15fps tetap dapat jendela yang mewakili
   pacing gerakan ~2 detik yang diasumsikan training). CNNsa-BiLSTM dijalankan atas ke-10nya;
   jendela dengan confidence top-1 tertinggi yang jadi jawaban akhir, seluruh 10 hasil tetap
   dicatat untuk layar Evaluasi.
4. MViTv2 memakai `ClipPreprocessor` (linspace sampling 16 frame dari seluruh rekaman, resize ke
   ukuran training) atas rekaman yang sama, dijalankan sekali per sesi.

### Aturan vector 126-float (WAJIB identik dengan training)

Diverifikasi dari `FeatureExtraction/Scripts/keypoint_extractor.py`,
`FeatureExtraction/Config/feature_config.py`, `Training/CNNsaBiLSTM/Models/cnnsa_bilstm.py`,
`Export/Scripts/export_cnnsa_bilstm.py` di repo riset ML.

- MediaPipe Hands, maksimal 2 tangan. Per tangan terdeteksi: 21 landmark × (x,y,z) = 63 nilai.
- Layout vector per frame (126 float): **tangan Kiri → [0:63], tangan Kanan → [63:126]**,
  tangan tidak terdeteksi = nol semua.
- **Normalisasi per tangan**: kurangi wrist (landmark index 0) dari seluruh 21 landmark, lalu bagi
  semua koordinat dengan jarak Euclidean(wrist, landmark index 9 = middle-finger MCP); skip
  pembagian jika jarak itu < 1e-6.
- Model input tensor **"keypoints"**, shape `(1,30,126)` float32. Output **"logits"** `(1,32)`
  (softmax + argmax → kelas + confidence).
- Mapping label → gloss BISINDO (`LabelMap.kt`, 32 kelas — daftar lengkap ada di kode, sumbernya
  disediakan terpisah oleh user karena `feature_config.py`'s `LABEL_MAP` di repo riset ML hanya
  placeholder `"label_N"`).

### Bug root-cause yang wajib diketahui sebelum menyentuh `HandLandmarkExtractor.kt`

Ditemukan 2026-08-20 dengan membandingkan `sequence.json` hasil ekspor 5 sesi live terhadap sample
training desktop asli untuk kelas gloss yang sama, lintas 5 signer training — dua bug yang saling
memperparah, **bukan** masalah kualitas model:

1. **Handedness terbalik**: label `categoryName()` MediaPipe live ("Left"/"Right") ternyata
   kebalikan total dari distribusi label di data training (mengganti ke kamera belakang, langkah
   perbaikan sebelumnya, TIDAK memperbaiki ini seperti yang diasumsikan). Fix: `"Left" -> 63,
   "Right" -> 0` (bukan pemetaan naif `"Left"->0`).
2. **Rotasi belum diterapkan**: bitmap sensor mentah yang diberikan ke MediaPipe di jalur
   CNNsa-BiLSTM tidak pernah dirotasi ke orientasi tegak sebelum `detectAsync()` — beda dari jalur
   MViTv2 yang sudah memanggil `OverlayRenderer.orientForDisplay()`. Fix: terapkan
   `OverlayRenderer.rotateNormalized()` (rotasi yang sama, sudah terverifikasi lewat overlay) ke
   tiap landmark (x,y) sebelum normalisasi wrist-relative.

Kedua fix divalidasi bersama secara offline (replay 5 sesi live sungguhan lewat model ONNX asli di
desktop): 0/5 benar → 5/5 benar, ~100% confidence masing-masing. Tervalidasi ulang di device
fisik pada pengujian lanjutan (4/5, lalu 3/3 dari sesi yang valid — 1 sesi lain sengaja direkam
dengan kamera digoyang sebagai uji ketahanan, bukan kegagalan). Detail lengkap, termasuk dua bug
tambahan yang lebih halus (flicker klasifikasi handedness saat gerakan cepat, dan misdetection
MediaPipe mengunci ke area bukan-tangan) serta mitigasinya (majority-vote smoothing per tangan
yang di-track, filter confidence handedness tambahan) ada di doc comment
`HandLandmarkExtractor.kt` dan `PROGRESS.md`.

## Struktur project

```
BISINDO_ANDROID_APP/
├── settings.gradle.kts, build.gradle.kts, gradle.properties
├── gradlew, gradlew.bat, gradle/wrapper/
├── PROGRESS.md                  log progress/keputusan lintas sesi
├── app/
│   ├── build.gradle.kts
│   ├── src/main/
│   │   ├── AndroidManifest.xml         permission CAMERA saja (tanpa INTERNET)
│   │   ├── assets/
│   │   │   ├── cnnsa_bilstm.onnx        copy manual dari repo riset ML
│   │   │   ├── mvit_v2.onnx(.data)      copy manual dari repo riset ML
│   │   │   └── hand_landmarker.task     dari bucket resmi MediaPipe
│   │   ├── res/{layout,values,xml}/
│   │   └── java/com/example/bisindorecognizer/
│   │       ├── MainActivity.kt                  layar Menu: Buka Kamera / Riwayat / Keluar
│   │       ├── CameraActivity.kt                layar Rekam & Bandingkan (lihat Fitur Utama)
│   │       ├── LabelMap.kt                      index (0-31) -> gloss BISINDO
│   │       ├── camera/AppCameraController.kt    CameraX ProcessCameraProvider + ImageAnalysis
│   │       ├── keypoints/HandLandmarkExtractor.kt    MediaPipe Tasks, vector 126-float,
│   │       │                                          koreksi handedness+rotasi (lihat di atas)
│   │       ├── keypoints/KeypointSequenceBuilder.kt  10-jendela geser + center-crop tunggal
│   │       ├── keypoints/ClipPreprocessor.kt         preprocessing klip 16-frame untuk MViTv2
│   │       ├── inference/OnnxPredictor.kt            sesi ONNX Runtime CNNsa-BiLSTM
│   │       ├── inference/MvitPredictor.kt            sesi ONNX Runtime MViTv2
│   │       ├── overlay/OverlayRenderer.kt            rotasi/orientasi, render komposit ke JPEG
│   │       ├── overlay/HandOverlayView.kt            gambar skeleton/bounding box live
│   │       ├── overlay/HandOverlayData.kt            data class hasil deteksi untuk overlay
│   │       ├── metrics/PerformanceMetrics.kt         snapshot CPU%/RAM sekitar tiap predict()
│   │       └── history/
│   │           ├── HistoryActivity.kt               daftar Riwayat
│   │           ├── HistoryPlaybackActivity.kt        flipbook + unduh sequence.json
│   │           ├── HistorySessionAdapter.kt          adapter RecyclerView Riwayat
│   │           ├── HistoryStore.kt                   simpan/baca sesi (files/history/)
│   │           ├── HistorySession.kt                 data class sesi + sessionNumbers()
│   │           ├── SequenceExporter.kt                ekspor sequence.json via share sheet
│   │           ├── EvaluationActivity.kt              tabel & grafik perbandingan
│   │           └── LatencyChartView.kt                grafik latensi custom view
└── README.md                    dokumen ini
```

### Dependency `app/build.gradle.kts`

```kotlin
implementation("androidx.core:core-ktx:1.15.0")
implementation("androidx.appcompat:appcompat:1.7.0")
implementation("com.google.android.material:material:1.12.0")
implementation("androidx.recyclerview:recyclerview:1.3.2")

val cameraxVersion = "1.6.1"
implementation("androidx.camera:camera-core:$cameraxVersion")
implementation("androidx.camera:camera-camera2:$cameraxVersion")
implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
implementation("androidx.camera:camera-view:$cameraxVersion")

implementation("com.google.mediapipe:tasks-vision:0.10.35")
implementation("com.microsoft.onnxruntime:onnxruntime-android:1.28.0")
implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
```
Versi di atas dipin 2026-07-31 dari metadata Google Maven/Maven Central saat itu, terbukti
kompatibel satu sama lain di device nyata. Kalau lanjut kerjakan repo ini jauh di kemudian hari,
cek ulang `maven-metadata.xml` sebelum upgrade — tidak wajib, tapi versi di atas bisa saja sudah
tidak "latest".

## Alur navigasi & lifecycle app

```
MainActivity (launcher, Menu)
 ├─ "Buka Kamera"  -> CameraActivity
 ├─ "Riwayat"       -> HistoryActivity -> (tap sesi) HistoryPlaybackActivity
 │                                     -> (tombol Evaluasi) EvaluationActivity
 └─ "Keluar"        -> finishAndRemoveTask()
```

**Tidak ada kamera nyala di background** setelah keluar/tutup kamera, dua lapis:
1. `AppCameraController` di-bind ke `lifecycleOwner = this` (CameraActivity) — CameraX otomatis
   unbind & lepas hardware kamera begitu lifecycle STOPPED/DESTROYED, termasuk kalau activity
   di-kill paksa sistem.
2. `CameraActivity.onDestroy()` eksplisit memanggil `cameraController.shutdown()`,
   `handLandmarkExtractor.close()`, `onnxPredictor.close()`, `mvitPredictor?.close()` — native
   resource MediaPipe & kedua sesi ONNX Runtime tidak otomatis terikat ke lifecycle activity
   seperti CameraX, jadi harus dilepas manual.

## Metrik performa (CPU% / RAM)

`metrics/PerformanceMetrics.kt` mengambil snapshot CPU dan RAM (PSS) tepat sebelum/sesudah
`predict()` tiap model, dipakai untuk mengisi kolom CPU%/RAM di layar Evaluasi dan menutup
komitmen di Bab 1 laporan skripsi ("penggunaan memori... serta konsumsi CPU") yang sebelumnya
tidak pernah benar-benar diukur. Nilai MViTv2 bernilai `null` (bukan nol) kalau MViTv2 gagal
dimuat/OOM/tidak ada frame untuk sesi itu — beda makna dari "diukur tapi hasilnya nol".

## Risiko / catatan teknis yang sudah divalidasi

1. **Konflik nama class**: wrapper CameraX sendiri dinamai `AppCameraController` (bukan
   `CameraController`) untuk menghindari ambiguitas dengan `androidx.camera.view.CameraController`.
2. **Disiplin `imageProxy.close()`**: wajib di `finally`, lupa bikin pipeline kamera macet diam-diam.
3. **Monotonicity timestamp `detectAsync`**: pakai `SystemClock.uptimeMillis()`, bukan
   `System.currentTimeMillis()` atau timestamp sensor `ImageProxy`.
4. **Threading**: callback hasil LIVE_STREAM MediaPipe tidak dijamin jalan di main thread — semua
   update `View` dipindah lewat `Handler(Looper.getMainLooper())`.
5. **Konflik dependency/native-lib**: `packaging{}` (excludes + `pickFirsts` pada
   `libc++_shared.so`) menangani konflik `META-INF/...` antar AAR CameraX/MediaPipe/ONNX Runtime;
   `abiFilters=["arm64-v8a"]` saja (ABI Poco M4 Pro).
6. **`compileSdk >= 36` untuk CameraX 1.6.1** — kejadian nyata, bukan teori: `compileSdk=35`
   gagal di `checkDebugAarMetadata`.
7. **`--` di tengah komentar XML itu ilegal** secara sintaks XML (cuma boleh sebagai penutup
   `-->`) — bikin `processDebugMainManifest`/manifest merger gagal parse tanpa pesan lokasi yang
   jelas. Semua komentar XML di repo ini sengaja menghindari `--` di tengah teks.
8. **Handedness + rotasi (root cause CNNsa-BiLSTM live)** — lihat bagian tersendiri di atas,
   temuan paling signifikan dalam pengembangan app ini.
9. **NeuQuant/GIF encoder bug** (fitur yang sudah dihapus): quantizer warna kustom untuk encoding
   GIF punya bug nyata (`unbiasnet()` lupa right-shift, index out of bounds) yang diperbaiki dan
   diverifikasi independen lewat port Python — tapi crash lain yang tidak berhasil didiagnosis
   (tanpa akses `adb logcat` saat itu) tetap terjadi persis di titik `startActivity()` share
   sheet. Setelah beberapa lapis mitigasi (`catch (e: Throwable)` untuk menangkap
   `OutOfMemoryError`, downscale bitmap, pause flipbook saat ekspor) tidak menyelesaikannya, fitur
   unduh/share GIF **dihapus total** (`gif/` package dihapus) — preview flipbook in-app yang tidak
   pernah memakai kode itu tetap ada dan tidak pernah bermasalah.

## Verifikasi

1. `.\gradlew.bat clean assembleDebug` -> `app\build\outputs\apk\debug\app-debug.apk`.
2. `adb devices` menampilkan device sebagai `device` (USB debugging authorized); `adb install -r`.
3. Test manual di device: Menu -> Buka Kamera -> izin kamera -> "Rekam (5 dtk)" -> lakukan satu
   gerakan isyarat -> hasil CNNsa-BiLSTM & MViTv2 (gloss, confidence, latency) tampil berdampingan,
   sesi tersimpan ke Riwayat. **Sudah tervalidasi berhasil berulang kali** di Poco M4 Pro,
   termasuk setelah perbaikan handedness+rotasi (sesi lanjutan: 4/5 lalu 3/3 benar dari sesi
   valid, confidence mendekati 100%) — lihat `PROGRESS.md` untuk log lengkap tiap sesi pengujian.
4. Layar Riwayat -> tap sesi -> flipbook frame terputar, tombol unduh data keypoint (JSON)
   membuka share sheet Android dengan `sequence.json` sesi itu.
5. Layar Evaluasi -> tabel menampilkan breakdown 10 jendela CNNsa-BiLSTM (satu ditandai
   "(dipilih)"), hasil MViTv2, latency, CPU%/RAM, ukuran model, dan grafik latensi untuk semua
   sesi tersimpan.
