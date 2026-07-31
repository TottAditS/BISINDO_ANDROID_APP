# BISINDO Live Sign-Recognition Android App

Aplikasi Android native (Kotlin) yang mengenali isyarat BISINDO secara real-time dari kamera live:
kamera -> ekstraksi keypoint tangan on-device (MediaPipe Hands) -> inference model ONNX
(CNNsa-BiLSTM) -> hasil prediksi ditampilkan di layar. Sepenuhnya offline, tidak ada permission
INTERNET.

Repo ini **terpisah total** dari repo riset ML Python (`CNN-LSTM_BISINDO_Android`) demi tetap
modular -- satu-satunya keterkaitan adalah 1 file model (`cnnsa_bilstm.onnx`) yang di-copy manual
ke `app/src/main/assets/`, tanpa dependency source code, submodule, atau shared build config.

## Status saat ini

App **sudah jalan end-to-end di device fisik** (Poco M4 Pro): scaffold selesai, build sukses,
diinstall manual, dan pipeline kamera -> keypoint -> inference -> tampilan prediksi terbukti
bekerja (tervalidasi lewat screenshot nyata, bukan cuma teori). Detail progress
per-langkah, keputusan/penyimpangan dari draft awal dokumen ini, dan rencana kerja berikutnya
dilacak terpisah di [`PROGRESS.md`](PROGRESS.md) -- baca itu dulu kalau lanjut sesi baru.

Ringkasan cepat:
- Toolchain, scaffold, build (`assembleDebug`), dan kedua file model di `assets/` -- **selesai**.
- App punya 2 layar: **Menu** (buka kamera / keluar) dan **Kamera** (preview + prediksi live +
  tombol tutup kamera) -- lihat bagian "Alur navigasi & lifecycle app" di bawah.
- Label prediksi ditampilkan sebagai **gloss BISINDO** (mis. "Saya"), bukan index mentah -- lihat
  tabel mapping di bagian preprocessing di bawah.
- **Belum dikerjakan** (scope sesi berikutnya): bounding box & skeleton overlay tangan di preview
  kamera, polish visual/tema, sanity check handedness/mirroring dengan gerakan yang diketahui
  artinya. MViTv2 **sengaja tidak diintegrasikan** (lihat bagian Context di bawah + `PROGRESS.md`
  untuk analisis kenapa).

## Context

Bagian riset ML (repo `CNN-LSTM_BISINDO_Android`) sudah selesai: PreProcessing -> FeatureExtraction
-> Training (CNNsa-BiLSTM 94.53% val acc, MViTv2 99.22%) -> Export ke ONNX, keduanya sudah
tervalidasi. Hanya **CNNsa-BiLSTM** (model ringan yang diusulkan, input keypoint MediaPipe 126-dim,
ONNX ~14.6MB) yang masuk ke aplikasi Android -- export ONNX MViTv2 (input raw video `(1,3,16,224,224)`,
~143MB termasuk external-data file) sengaja tidak diintegrasikan, terlalu berat untuk mobile CPU-only
(lihat analisis lengkap di `PROGRESS.md`).

Tujuan: aplikasi demo yang benar-benar jalan -- kamera live -> ekstraksi keypoint tangan on-device
-> inference model -> tampilkan hasil prediksi di layar.

**Batasan eksplisit:**
- **Tidak install Android Studio** (IDE berat). Semua dilakukan dari **VSCode + terminal**, cukup
  JDK + Android SDK command-line tools + Gradle -- alur kerja yang valid dan ringan.
- **Kotlin**, Android native (bukan Unity -- MediaPipe Hands tidak didukung resmi di Unity, hanya
  plugin komunitas yang rapuh; Kotlin dapat dukungan resmi penuh dari MediaPipe Tasks + ONNX Runtime).
- **Repo/folder terpisah total** dari repo riset ML Python.
- **Input real-time via kamera** (bukan upload video).
- Device testing: HP fisik, **Poco M4 Pro** (Android 11+), via USB debugging -- tidak perlu
  emulator (lebih ringan).
- Scope untuk tahap ini: **proof-of-concept minimal** -- kamera live, label prediksi + confidence
  ditampilkan, dengan navigasi menu dasar (buka/tutup kamera, keluar app). Fokus buktikan model
  ONNX jalan benar on-device. Visual masih sederhana, belum ada bounding box/skeleton overlay.

## Toolchain (tanpa Android Studio)

- **JDK 17**
- **Gradle 8.13** (download sekali sebagai binary biasa, dipakai cuma untuk bootstrap wrapper)
- **Android SDK "Command line tools only"** (dari developer.android.com), lalu via `sdkmanager`:
  `platform-tools` (adb) + `platforms;android-35` + `platforms;android-36` +
  `build-tools;35.0.0` + `build-tools;36.0.0`
- **AGP 8.13.0** + **Kotlin plugin 2.1.20** (pasangan versi yang terdokumentasi kompatibel;
  butuh JDK 17)
- `compileSdk=36`, `targetSdk=34`, `minSdk=26`. **`compileSdk` dinaikkan dari draft awal 35 ke
  36** karena `androidx.camera:*:1.6.1` mensyaratkan compileSdk >= 36 (ketauan pas
  `assembleDebug` gagal di task `checkDebugAarMetadata` -- lihat Risiko #12). `targetSdk`/
  `minSdk` tidak berubah, jadi kompatibilitas device (Poco M4 Pro Android 11/API 30) tidak
  terpengaruh -- compileSdk cuma menentukan API yang boleh dipakai saat compile.
- Testing: Poco M4 Pro. **Diputuskan pakai install manual (sideload APK lewat file manager di
  HP), bukan `adb install`** -- konsekuensinya tidak ada `adb logcat` buat lihat crash detail
  kecuali USB debugging disambungkan belakangan untuk sesi debugging.

### VSCode extensions

Build/run sepenuhnya lewat terminal (`.\gradlew.bat assembleDebug`, `adb install -r`); extension
di bawah ini murni untuk editing.

| Extension | ID | Fungsi |
|---|---|---|
| Kotlin | `fwcd.kotlin` | Syntax highlighting + language server (autocomplete, go-to-definition, error inline) untuk file `.kt`. |
| Gradle for Java | `vscjava.vscode-gradle` | Panel sidebar untuk lihat & jalankan Gradle task (`assembleDebug`, `tasks`, dll) tanpa hafal command. |
| XML | `redhat.vscode-xml` | Syntax highlight + validasi untuk `AndroidManifest.xml` dan layout XML. |

## Cara kerja preprocessing real-time (PENTING -- beda dari pipeline training)

Training pipeline (Python) memproses video yang SUDAH direkam secara batch (PreProcessing lalu
FeatureExtraction). Untuk aplikasi live camera, **tidak ada langkah preprocessing terpisah** --
logika yang sama (ekstraksi + normalisasi keypoint) dijalankan terus-menerus per frame sebagai
bagian dari loop aplikasi:

1. Kamera menangkap frame terus-menerus (~15-30 fps).
2. **Tiap frame langsung diproses saat itu juga**: MediaPipe extract landmark tangan -> dihitung
   jadi vector 126 angka (lihat aturan normalisasi di bawah). Ini menggantikan
   `keypoint_extractor.py`, tapi jalan real-time bukan batch.
3. Vector 126-angka itu masuk ke **rolling buffer** yang menyimpan 30 frame terakhir -- begitu
   penuh, itu langsung jadi "sequence" `(30,126)` input model, terus ter-update tiap frame baru
   masuk (frame paling lama otomatis dibuang). Ini menggantikan padding/cropping yang dipakai
   `sequence_builder.py` untuk video training -- di app selalu tepat T=30, jadi logika pad/crop
   tidak diperlukan sama sekali.
4. Begitu buffer penuh, setiap beberapa frame model ONNX dijalankan atas window saat itu ->
   hasil prediksi ditampilkan.

## Preprocessing yang WAJIB direplikasi persis di Kotlin

Diverifikasi dengan membaca langsung: `FeatureExtraction/Scripts/keypoint_extractor.py`,
`FeatureExtraction/Config/feature_config.py`, `Training/CNNsaBiLSTM/Models/cnnsa_bilstm.py`,
`Export/Scripts/export_cnnsa_bilstm.py` di repo riset ML.

- MediaPipe Hands, maksimal 2 tangan. Per tangan terdeteksi: 21 landmark x (x,y,z) = 63 nilai.
- Layout vector per frame (126 float): **tangan Kiri -> [0:63], tangan Kanan -> [63:126]**
  (berdasarkan label handedness MediaPipe), tangan tidak terdeteksi = nol semua.
- **Normalisasi per tangan**: kurangi wrist (landmark index 0) dari seluruh 21 landmark, lalu bagi
  semua koordinat dengan jarak Euclidean(wrist, landmark index 9 = middle-finger MCP); skip
  pembagian jika jarak itu < 1e-6. Ini membuatnya invariant terhadap posisi & ukuran tangan.
- Model input tensor bernama **"keypoints"**, shape `(1,30,126)` float32. Output:
  **"logits"** `(1,32)` (softmax + argmax -> kelas prediksi + confidence) dan "attn_weights"
  `(1,30,1)` (abaikan untuk app).
- File model yang perlu di-bundle: **copy manual** `Export/Output/onnx/cnnsa_bilstm.onnx`
  (~14.6MB terkonfirmasi, README draft awal memperkirakan ~44MB -- ternyata lebih kecil, 1 file
  saja, tidak butuh file `.data` eksternal) dari repo riset ML ke project Android ini.
- **Mapping label -> gloss BISINDO**: `FeatureExtraction/Config/feature_config.py` baris 97
  mendefinisikan `LABEL_MAP = {i: f"label_{i}" ...}` -- placeholder ini **tidak** berisi gloss
  asli, dan tidak ada mapping kata/gloss di file config manapun di repo riset ML (nama file video
  training di `Datasets/WLBisindo/` juga literal `signer0_labelN_sampleM.mp4`). Mapping di bawah
  disediakan terpisah oleh user (bukan dari repo riset ML) dan dipakai di `LabelMap.kt` untuk
  ditampilkan di app, menggantikan `"label_<idx>"` mentah:

  | Label | Gloss | English | | Label | Gloss | English |
  |---|---|---|---|---|---|---|
  | 0 | Air | Water | | 16 | Mengapa | Why |
  | 1 | Belajar | Learn | | 17 | Bagaimana | How |
  | 2 | Cari | Search | | 18 | Merah | Red |
  | 3 | Hari | Day | | 19 | Kuning | Yellow |
  | 4 | Ingat | Remember | | 20 | Hijau | Green |
  | 5 | Lagi | Again | | 21 | Hitam | Black |
  | 6 | Maaf | Sorry | | 22 | Dengar | Hear |
  | 7 | Makan | Eat | | 23 | Berangkat | Depart |
  | 8 | Motor | Motorcycle | | 24 | Datang | Come |
  | 9 | Saya | I | | 25 | Teman | Friend |
  | 10 | Terima kasih | Thank you | | 26 | Keluarga | Family |
  | 11 | Tuli | Deaf | | 27 | Rumah | House |
  | 12 | Apa | What | | 28 | Pagi | Morning |
  | 13 | Siapa | Who | | 29 | Siang | Noon |
  | 14 | Kapan | When | | 30 | Sore | Afternoon |
  | 15 | Di mana | Where | | 31 | Malam | Night |
- **Catatan mirroring**: label Left/Right MediaPipe berdasarkan frame sensor mentah. Kalau app
  pakai kamera depan (mirrored dari sudut pandang user), assignment Left/Right -> setengah-vector
  bisa terbalik dibanding cara video training direkam. **Wajib divalidasi empiris** di run pertama
  (tunjukkan satu tangan yang diketahui, pastikan masuk ke setengah vector yang tepat) sebelum
  percaya hasil prediksi multi-tangan.

## Struktur project: `BISINDO_ANDROID_APP/` (repo terpisah)

```
BISINDO_ANDROID_APP/
├── settings.gradle.kts, build.gradle.kts, gradle.properties (android.useAndroidX=true -- gampang
│                                                              lupa karena tidak ada wizard yang set)
├── gradlew, gradlew.bat, gradle/wrapper/     dibuat sekali lewat `gradle wrapper --gradle-version 8.13`
├── PROGRESS.md                  log progress/keputusan lintas sesi -- baca ini duluan kalau lanjut kerja
├── app/
│   ├── build.gradle.kts        dependency di bawah; blok packaging{} untuk hindari konflik
│   │                           META-INF antar AAR CameraX/MediaPipe/ONNX Runtime;
│   │                           abiFilters=["arm64-v8a"] saja
│   ├── src/main/
│   │   ├── AndroidManifest.xml         permission CAMERA saja (tanpa INTERNET -- full offline);
│   │   │                               2 activity: MainActivity (launcher/menu) + CameraActivity
│   │   ├── assets/
│   │   │   ├── cnnsa_bilstm.onnx       COPY MANUAL dari Export/Output/onnx/ (repo riset ML)
│   │   │   └── hand_landmarker.task    download dari model-zoo bucket resmi MediaPipe
│   │   ├── res/
│   │   │   ├── layout/activity_main.xml     layar Menu: judul + tombol "Buka Kamera"/"Keluar"
│   │   │   ├── layout/activity_camera.xml   layar Kamera: PreviewView + tombol "Tutup Kamera" +
│   │   │   │                                TextView overlay hasil prediksi
│   │   │   ├── values/{strings,colors,themes}.xml
│   │   │   ├── drawable/ic_launcher_foreground.xml   launcher icon (vector, tanpa PNG)
│   │   │   └── mipmap-anydpi-v26/ic_launcher.xml     adaptive icon (aman karena minSdk=26 pas
│   │   │                                              dengan syarat minimum adaptive icon)
│   │   └── java/com/example/bisindorecognizer/
│   │       ├── MainActivity.kt                 layar Menu (launcher) -- "Buka Kamera" buka
│   │       │                                   CameraActivity, "Keluar" -> finishAndRemoveTask()
│   │       ├── CameraActivity.kt               layar Kamera -- rangkai semua komponen di bawah,
│   │       │                                   pindahkan callback thread MediaPipe -> main thread
│   │       │                                   untuk update UI text, lepas semua resource native
│   │       │                                   di onDestroy() (lihat bagian navigasi & lifecycle)
│   │       ├── LabelMap.kt                     index (0-31) -> gloss BISINDO, dipakai
│   │       │                                   CameraActivity buat tampilkan gloss bukan index
│   │       ├── camera/AppCameraController.kt   CameraX ProcessCameraProvider + ImageAnalysis
│   │       │                                   (OUTPUT_IMAGE_FORMAT_RGBA_8888 -> imageProxy.toBitmap())
│   │       ├── keypoints/HandLandmarkExtractor.kt   MediaPipe Tasks HandLandmarker, mode
│   │       │                                        LIVE_STREAM, bangun vector 126-float sesuai
│   │       │                                        aturan di atas
│   │       ├── keypoints/SequenceBuffer.kt     rolling window tersinkronisasi, 30 vector
│   │       └── inference/OnnxPredictor.kt      session ONNX Runtime, softmax+argmax atas "logits"
└── README.md                    dokumen ini
```

### Dependency `app/build.gradle.kts` (versi terkonfirmasi jalan -- lihat catatan)
```
androidx.camera:camera-{core,camera2,lifecycle,view}:1.6.1   (butuh compileSdk >= 36)
com.google.mediapipe:tasks-vision:0.10.35      (dipin 2026-07-31 dari metadata Google Maven --
                                                 versi patch 0.10.x terbaru saat itu; tidak crash
                                                 di createFromOptions() saat run pertama di Poco
                                                 M4 Pro/API 30)
com.microsoft.onnxruntime:onnxruntime-android:1.28.0   (dipin 2026-07-31, latest stable per
                                                 Maven Central metadata saat itu)
org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0
```
Catatan: kalau lanjut kerjakan repo ini jauh di kemudian hari, versi-versi di atas mungkin sudah
tidak "latest" lagi -- cek ulang `maven-metadata.xml` di `dl.google.com`/`repo1.maven.org` kalau
mau upgrade, tapi tidak wajib (versi di atas sudah terbukti kompatibel satu sama lain).

Root `build.gradle.kts`:
```kotlin
plugins {
    id("com.android.application") version "8.13.0" apply false
    id("org.jetbrains.kotlin.android") version "2.1.20" apply false
}
```

`settings.gradle.kts`:
```kotlin
pluginManagement {
    repositories { google(); mavenCentral(); gradlePluginPortal() }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories { google(); mavenCentral() }
}
rootProject.name = "BisindoRecognizer"
include(":app")
```

`app/build.gradle.kts` (kerangka lengkap):
```kotlin
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.bisindorecognizer"
    compileSdk = 36  // 35 di draft awal gagal build -- CameraX 1.6.1 butuh compileSdk >= 36

    defaultConfig {
        applicationId = "com.example.bisindorecognizer"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        ndk { abiFilters += listOf("arm64-v8a") }
    }

    buildFeatures { viewBinding = true }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    packaging {
        resources {
            excludes += setOf(
                "META-INF/DEPENDENCIES", "META-INF/LICENSE*", "META-INF/NOTICE*",
                "META-INF/*.kotlin_module", "META-INF/INDEX.LIST"
            )
        }
        jniLibs { pickFirsts += setOf("**/libc++_shared.so") }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")

    val cameraxVersion = "1.6.1"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    implementation("com.google.mediapipe:tasks-vision:0.10.35")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.28.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
}
```

`AndroidManifest.xml` inti:
```xml
<uses-permission android:name="android.permission.CAMERA"/>
<uses-feature android:name="android.hardware.camera.any" android:required="true"/>
<!-- Tanpa INTERNET permission: semuanya jalan offline/on-device -->
```
**Jebakan XML**: komentar di atas SENGAJA pakai `:` bukan `--` di tengah teks -- draft awal
dokumen ini pakai `--` (mis. `<!-- A -- B -->`) yang ternyata **ilegal secara sintaks XML**
(comment tidak boleh mengandung `--` di tengah, cuma boleh sebagai penutup `-->`). Ini bikin
`processDebugMainManifest` gagal parse manifest -- ditemukan & diperbaiki lewat build asli, lihat
Risiko #13.

## Alur navigasi & lifecycle app (Menu <-> Kamera <-> Keluar)

App terdiri dari **2 activity**, bukan 1 layar langsung buka kamera seperti draft awal dokumen
ini -- ditambahkan supaya user bisa keluar/tutup kamera secara eksplisit tanpa was-was kamera
tetap menyala di background:

```
MainActivity (launcher, layar Menu)
  |
  |-- tombol "Buka Kamera"  -->  startActivity(CameraActivity)
  |
  '-- tombol "Keluar"       -->  finishAndRemoveTask()  (app benar-benar hilang dari recent apps)

CameraActivity (layar Kamera, dibuka dari Menu)
  |
  |-- tombol "Tutup Kamera" atau tombol Back sistem  -->  finish()  -->  balik ke MainActivity
  |
  '-- (activity mana pun yang finish selalu memicu onDestroy())
```

**Jaminan tidak ada kamera nyala di background setelah keluar/tutup kamera** -- dua lapis:
1. `AppCameraController` di-bind ke `lifecycleOwner = this` (CameraActivity) lewat
   `cameraProvider.bindToLifecycle(...)`. Ini bagian dari desain CameraX sendiri: begitu
   lifecycle activity itu STOPPED/DESTROYED, CameraX **otomatis** unbind & lepas hardware
   kamera -- tidak perlu kode manual untuk ini, terjadi walau activity di-kill paksa oleh sistem
   (mis. swipe dari recent apps) sekalipun `onDestroy()` sempat tidak jalan penuh.
2. `CameraActivity.onDestroy()` secara eksplisit memanggil `cameraController.shutdown()`
   (matikan executor thread kamera), `handLandmarkExtractor.close()`, dan `onnxPredictor.close()`
   -- ini melepas native resource MediaPipe HandLandmarker & sesi ONNX Runtime, yang **tidak**
   otomatis terikat ke lifecycle activity seperti CameraX (kalau tidak di-close manual, native
   memory-nya bisa nyangkut walau UI activity sudah tidak terlihat).

Catatan realistis: Android tidak menjamin proses aplikasi langsung mati 100% dari RAM begitu
task di-remove (`finishAndRemoveTask()`) -- itu keputusan OS (bisa saja proses masih ada sebentar
dalam status "cached" tanpa activity/resource aktif apa pun, lalu direclaim kapan saja). Yang
**bisa** dan **sudah** dijamin oleh kode di atas: tidak ada hardware kamera yang tetap aktif,
tidak ada thread/native session yang masih jalan setelah CameraActivity ditutup -- itu yang
sebenarnya penting (baterai tidak boros, LED kamera mati, tidak ada CPU usage tersisa).

## Detail kode per file (level API, siap implementasi)

### `camera/AppCameraController.kt`
Pakai `ProcessCameraProvider` + `Preview`/`ImageAnalysis` eksplisit (BUKAN
`androidx.camera.view.CameraController` -- nama class beda, dan kurang fleksibel untuk format
RGBA yang dibutuhkan):
```kotlin
val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
cameraProviderFuture.addListener({
    val cameraProvider = cameraProviderFuture.get()
    val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
    val imageAnalysis = ImageAnalysis.Builder()
        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888) // supaya imageProxy.toBitmap() bisa dipakai
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .build()
    imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
        try {
            val bitmap = imageProxy.toBitmap()
            val rotation = imageProxy.imageInfo.rotationDegrees
            onFrame(bitmap, rotation, SystemClock.uptimeMillis())
        } finally {
            imageProxy.close() // WAJIB selalu close, kalau tidak pipeline macet setelah beberapa frame
        }
    }
    cameraProvider.unbindAll()
    cameraProvider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_FRONT_CAMERA, preview, imageAnalysis)
}, ContextCompat.getMainExecutor(context))
```
`ImageProxy.toBitmap()` tersedia sejak CameraX 1.3+, hanya jalan kalau output format di-set
`RGBA_8888` secara eksplisit -- ini trik supaya tidak perlu konversi manual YUV_420_888 -> RGB.
Pakai `Executors.newSingleThreadExecutor()` khusus untuk `cameraExecutor`, jangan main executor.

### `keypoints/HandLandmarkExtractor.kt`
```kotlin
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage

class HandLandmarkExtractor(context: Context, private val onVector: (FloatArray) -> Unit) {
    private val handLandmarker: HandLandmarker
    private val busy = AtomicBoolean(false)

    init {
        val baseOptions = BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build()
        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumHands(2)
            .setMinHandDetectionConfidence(0.5f)
            .setMinHandPresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setResultListener(::onResult)
            .setErrorListener { e -> Log.e("HandLandmarker", "error", e) }
            .build()
        handLandmarker = HandLandmarker.createFromOptions(context, options)
    }

    fun submit(bitmap: Bitmap, timestampMs: Long) {
        if (!busy.compareAndSet(false, true)) return // drop frame kalau masih proses yang sebelumnya
        handLandmarker.detectAsync(BitmapImageBuilder(bitmap).build(), timestampMs)
    }

    private fun onResult(result: HandLandmarkerResult, input: MPImage) {
        busy.set(false)
        onVector(buildVector126(result))
    }

    private fun buildVector126(result: HandLandmarkerResult): FloatArray {
        val vector = FloatArray(126)
        for (i in result.landmarks().indices) {
            val label = result.handedness()[i][0].categoryName() // "Left" atau "Right"
            val offset = if (label == "Left") 0 else 63
            val lm = result.landmarks()[i]
            val wrist = lm[0]; val midMcp = lm[9]
            val dx = midMcp.x() - wrist.x(); val dy = midMcp.y() - wrist.y(); val dz = midMcp.z() - wrist.z()
            val scale = sqrt(dx*dx + dy*dy + dz*dz)
            for (j in 0 until 21) {
                var x = lm[j].x() - wrist.x(); var y = lm[j].y() - wrist.y(); var z = lm[j].z() - wrist.z()
                if (scale >= 1e-6f) { x /= scale; y /= scale; z /= scale }
                vector[offset + j*3] = x; vector[offset + j*3 + 1] = y; vector[offset + j*3 + 2] = z
            }
        }
        return vector
    }
}
```
Catatan penting yang sudah dibakukan di desain ini:
- `timestampMs` di `detectAsync` HARUS strictly increasing -- pakai `SystemClock.uptimeMillis()`
  yang diambil saat frame dianalisis, JANGAN `System.currentTimeMillis()` (bisa lompat karena clock
  sync) dan JANGAN sensor timestamp dari `ImageProxy` (basis clock/unit beda, bisa melanggar
  monotonicity dan throw error).
- Flag `busy` mencegah `detectAsync` dipanggil lebih cepat dari kemampuan MediaPipe memproses --
  `STRATEGY_KEEP_ONLY_LATEST` di `ImageAnalysis` cuma mengatur frame yang sampai ke analyzer, tidak
  tahu apakah panggilan async HandLandmarker sebelumnya sudah selesai.

### `keypoints/SequenceBuffer.kt`
```kotlin
class SequenceBuffer(private val windowSize: Int = 30, private val featureSize: Int = 126) {
    private val buffer = ArrayDeque<FloatArray>(windowSize)
    @Synchronized fun push(vector: FloatArray) {
        if (buffer.size == windowSize) buffer.removeFirst()
        buffer.addLast(vector)
    }
    @Synchronized fun isFull() = buffer.size == windowSize
    @Synchronized fun toFlatFloatArray(): FloatArray {
        val out = FloatArray(windowSize * featureSize)
        buffer.forEachIndexed { i, v -> System.arraycopy(v, 0, out, i * featureSize, featureSize) }
        return out
    }
}
```

### `inference/OnnxPredictor.kt`
```kotlin
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import java.nio.FloatBuffer
import kotlin.math.exp

class OnnxPredictor(context: Context) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(
        context.assets.open("cnnsa_bilstm.onnx").use { it.readBytes() },
        OrtSession.SessionOptions()
    )

    fun predict(sequence: FloatArray): Pair<Int, Float> {
        val shape = longArrayOf(1, 30, 126)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(sequence), shape).use { inputTensor ->
            session.run(mapOf("keypoints" to inputTensor), setOf("logits")).use { results ->
                val logits = (results.get("logits").get().value as Array<FloatArray>)[0]
                val probs = softmax(logits)
                val idx = probs.indices.maxByOrNull { probs[it] }!!
                return idx to probs[idx]
            }
        }
    }

    private fun softmax(x: FloatArray): FloatArray {
        val m = x.max()
        val exps = FloatArray(x.size) { exp((x[it] - m).toDouble()).toFloat() }
        val sum = exps.sum()
        return FloatArray(exps.size) { exps[it] / sum }
    }

    fun close() { session.close() }
}
```
Catatan:
- `env.createSession(byte[], SessionOptions)` adalah cara resmi memuat dari Android assets (tidak
  perlu copy ke file sementara). Ada known bug ONNX Runtime Android (GitHub issue #19599) hanya
  muncul kalau model berukuran ratusan MB ke atas dan bikin heap JVM penuh -- model ~14.6MB kita
  jauh di bawah itu; kalau nanti ganti model jauh lebih besar (mis. MViTv2 ~143MB dengan
  external-data file, lihat `PROGRESS.md`), **wajib** copy asset ke `context.filesDir` dulu lalu
  pakai `env.createSession(filePath, options)` -- `createSession(byte[])` tidak bisa resolve file
  `.onnx.data` eksternal sama sekali, bukan cuma soal performa.
- `OrtSession.Result.get(String)` return `java.util.Optional<OnnxValue>` -- harus `.get()` dulu
  sebelum cast ke `OnnxTensor`.
- `OnnxTensor` dan `OrtSession.Result` implement `AutoCloseable` -- selalu bungkus `.use{}` supaya
  tidak leak native memory (penting karena `predict()` dipanggil berulang di loop live).

### `LabelMap.kt`
Object statis, satu fungsi `gloss(index: Int): String` yang me-return gloss BISINDO dari array
32 string (urutan sesuai tabel mapping di bagian preprocessing di atas), fallback ke
`"label_$index"` kalau index di luar 0-31 (harusnya tidak pernah terjadi karena output model
tetap `(1,32)`, tapi jaga-jaga).

### `MainActivity.kt` (layar Menu, launcher)
- Extends `AppCompatActivity`, inflate `activity_main.xml` (judul app + 2 tombol).
- Tidak menyentuh kamera/permission sama sekali -- murni navigasi.
- `btnOpenCamera.setOnClickListener { startActivity(Intent(this, CameraActivity::class.java)) }`
- `btnQuitApp.setOnClickListener { finishAndRemoveTask() }`

### `CameraActivity.kt` (layar Kamera)
- Extends `AppCompatActivity`, inflate `activity_camera.xml` (PreviewView + TextView hasil +
  tombol Tutup Kamera).
- `registerForActivityResult(ActivityResultContracts.RequestPermission())` untuk `CAMERA`; setelah
  granted, baru rangkai camera + extractor + buffer + predictor. Diminta di activity ini (bukan
  di Menu) supaya user tidak ditanya izin kamera sebelum benar-benar mau pakai kamera.
- `btnCloseCamera.setOnClickListener { finish() }` -- balik ke Menu, memicu `onDestroy()`.
- **Aturan threading**: callback hasil LIVE_STREAM MediaPipe jalan di thread internal MediaPipe
  (bukan jaminan main thread atau thread executor CameraX) -- jalankan `SequenceBuffer.push()` +
  `OnnxPredictor.predict()` langsung di thread itu (cukup ringan), tapi pindah ke main thread HANYA
  untuk update `TextView.text = ...`, misal lewat `Handler(Looper.getMainLooper()).post { }`.
- `onDestroy()`: `cameraController.shutdown()` + `handLandmarkExtractor.close()` +
  `onnxPredictor.close()` -- lihat bagian "Alur navigasi & lifecycle app" untuk kenapa ini perlu
  eksplisit (tidak otomatis seperti CameraX).

## Langkah CLI scaffold -> build -> install (mesin Windows bersih)

1. Install **JDK 17** (Temurin/Adoptium), cek `java -version`.
2. Download **Gradle 8.13** binary-only zip dari gradle.org/releases, extract ke misal
   `C:\gradle\gradle-8.13`, tambahkan `bin` ke `PATH`.
3. Download **Android SDK "Command line tools only"** untuk Windows dari
   `https://developer.android.com/studio#command-line-tools-only`. Extract, lalu susun sesuai
   aturan Google: folder `cmdline-tools` dari dalam zip harus dipindah ke dalam folder bernama
   `latest` (`<sdk_root>\cmdline-tools\latest\...`).
4. Set `ANDROID_HOME` (atau `ANDROID_SDK_ROOT`) ke `<sdk_root>`; tambahkan
   `<sdk_root>\cmdline-tools\latest\bin` dan `<sdk_root>\platform-tools` ke `PATH`.
5. `sdkmanager --licenses` (terima semua), lalu:
   `sdkmanager "platform-tools" "platforms;android-35" "build-tools;35.0.0"`, dan (dibutuhkan
   belakangan karena CameraX 1.6.1 minta compileSdk >= 36, lihat Risiko #12) tambahan:
   `sdkmanager "platforms;android-36" "build-tools;36.0.0"`
6. Buat struktur project di root repo ini (`BISINDO_ANDROID_APP/`) dengan
   `app/src/main/{java,res,assets}` dst.
7. Tulis `settings.gradle.kts`, root `build.gradle.kts`, `app/build.gradle.kts`,
   `gradle.properties`, `AndroidManifest.xml`, layout XML, dan file Kotlin sesuai desain di atas.
8. Letakkan `hand_landmarker.task` (download dari model-zoo MediaPipe -- lihat catatan sumber di
   bawah) dan `cnnsa_bilstm.onnx` (copy manual dari repo riset ML, ~14.6MB) ke `app/src/main/assets/`.
9. Dari root project `BISINDO_ANDROID_APP/`, bootstrap wrapper pakai Gradle binary yang sudah
   diinstall manual (sekali saja):
   `gradle wrapper --gradle-version 8.13`
   Ini generate `gradlew`, `gradlew.bat`, `gradle/wrapper/gradle-wrapper.properties` -- selanjutnya
   pakai `.\gradlew.bat` (PowerShell) saja; Gradle standalone tidak dipakai lagi setelah ini.
10. Smoke test: `.\gradlew.bat tasks` (memaksa download/resolve pertama kali AGP, plugin Kotlin,
    semua dependency Maven -- lakukan ini SEBELUM nulis semua kode app, supaya masalah resolusi
    dependency terpisah dari masalah kode app).
11. Build: `.\gradlew.bat assembleDebug` -> hasil `app\build\outputs\apk\debug\app-debug.apk`.
12. **Install ke device** -- dua opsi:
    - **Lewat adb** (perlu USB debugging): aktifkan Developer Options -> USB debugging di Poco
      M4 Pro, sambung USB, authorize prompt RSA key, `adb devices` pastikan muncul sebagai
      `device` (bukan `unauthorized`), lalu `adb install -r app\build\outputs\apk\debug\app-debug.apk`.
    - **Manual sideload** (dipakai di project ini): pindahkan `app-debug.apk` ke HP (USB file
      transfer/lainnya), install langsung dari file manager di HP (izinkan "install dari sumber
      ini" kalau diblokir). Lebih simpel, tapi konsekuensinya tidak ada `adb logcat` buat lihat
      crash detail kecuali USB debugging disambungkan belakangan.
13. Buka app: tap icon (masuk ke layar Menu), tap "Buka Kamera" untuk masuk ke layar Kamera --
    atau langsung lewat adb: `adb shell am start -n com.example.bisindorecognizer/.MainActivity`.
14. Untuk debug tanpa Logcat window Android Studio (butuh adb tersambung): `adb logcat *:E` atau
    filter by tag -- ini satu-satunya cara lihat crash kalau install manual.

## Sumber file yang perlu di-copy/download (bukan dari repo ini)

Repo ini butuh 2 file eksternal untuk masuk ke `app/src/main/assets/`, dari 2 sumber berbeda:

- **Dari repo riset ML** (`CNN-LSTM_BISINDO_Android`) -- copy manual:
  `Export/Output/onnx/cnnsa_bilstm.onnx` -> `app/src/main/assets/cnnsa_bilstm.onnx`
- **Dari bucket resmi MediaPipe** (bukan dari repo riset ML) -- download langsung:
  `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`
  -> `app/src/main/assets/hand_landmarker.task`
  (atau pin versi `float16/1/hand_landmarker.task` untuk reproducibility)

Selain 1 file model di atas, repo ini tidak punya dependency lain ke repo riset ML -- tidak ada
import source code, submodule, atau shared build config.

## Risiko / hal yang perlu diwaspadai (sudah divalidasi riset)

1. **Konflik nama class**: `androidx.camera.view.CameraController` adalah class asli CameraX.
   Class wrapper sendiri dinamai `AppCameraController` (bukan `CameraController`) untuk menghindari
   ambiguitas kalau ada wildcard import `androidx.camera.*`.
2. **Disiplin `imageProxy.close()`**: lupa close di `finally` (misal ada exception di tengah
   `analyze()`) bikin seluruh pipeline kamera macet diam-diam setelah beberapa frame -- bug pemula
   CameraX yang paling umum.
3. **Monotonicity timestamp `detectAsync`**: harus strictly increasing; clock source yang salah
   akan throw error saat runtime.
4. **Threading**: callback hasil LIVE_STREAM tidak dijamin jalan di executor analysis atau main
   thread -- jangan pernah sentuh View langsung dari situ.
5. **Mirroring kamera depan vs handedness**: klasifikasi Left/Right MediaPipe berdasarkan frame
   sensor mentah. Kalau pakai kamera depan (selfie), frame biasanya di-mirror relatif terhadap apa
   yang dilihat user di layar. Kalau pipeline training Python diberi frame yang TIDAK di-mirror,
   assignment Left/Right -> setengah-vector saat inference bisa terbalik sistematis dibanding
   training. **Wajib divalidasi empiris di awal** (tunjukkan satu tangan yang diketahui, pastikan
   masuk ke setengah 0:63 atau 63:126 yang diharapkan) sebelum percaya prediksi multi-tangan.
6. **Konflik dependency/native-lib**: menggabungkan CameraX + MediaPipe Tasks + ONNX Runtime AAR
   adalah sumber umum error build `More than one file was found with OS independent path
   META-INF/...`, dan kadang runtime `UnsatisfiedLinkError: libmediapipe_tasks_vision_jni.so not
   found`. Blok `packaging{}` di atas (excludes + `pickFirsts` pada `libc++_shared.so`) menangani
   error build; kalau kena `UnsatisfiedLinkError` saat runtime, cek jangan sampai ada artifact
   legacy `com.google.mediapipe:solution-core`/`mediapipe-solution-*` ikut ter-pull bareng
   `tasks-vision` (penyebab known conflict per MediaPipe issue #5717), dan pastikan `abiFilters`
   sudah include `arm64-v8a` (ABI Poco M4 Pro).
7. **Known crash versi spesifik MediaPipe**: ada open GitHub issue soal crash di
   `HandLandmarker.createFromOptions()` pada device API 29 dengan tasks-vision 0.10.26. Target kita
   API 30, cukup dekat untuk perlu di-test segera setelah build pertama -- siap coba versi 0.10.x
   patch yang berdekatan kalau `createFromOptions()` crash saat launch pertama.
8. **Akses output ONNX Runtime via `Optional`**: `results.get("logits")` return
   `java.util.Optional<OnnxValue>` -- harus `.get()` dulu sebelum cast, gampang kelewat kalau
   terbiasa akses map Kotlin biasa.
9. **Versi `onnxruntime-android` belum pasti**: riset menghasilkan angka "latest" yang tidak
   konsisten antar mirror Maven (1.22.0 sampai 1.28.0 semua muncul). Jangan percaya satu angka --
   cek langsung `central.sonatype.com` tepat sebelum build.
10. **Performa di Poco M4 Pro (MediaTek Helio G96, Cortex-A75/A55, tanpa NPU yang bisa diandalkan)**:
    menjalankan capture CameraX + HandLandmarker (CPU) + ONNX Runtime (CPU) bersamaan di frame rate
    penuh bisa berat. Rekomendasi: resolusi target `ImageAnalysis` moderat, guard frame-drop `busy`
    yang sudah ada di desain extractor, dan throttle inference ONNX supaya jalan tiap beberapa frame
    saja (bukan tiap kali buffer ke-push).
11. **`android.useAndroidX=true`** di `gradle.properties` tidak di-set otomatis oleh wizard apapun
    di project CLI-from-scratch -- kalau kelewat, build langsung gagal karena semua dependency di
    sini berbasis AndroidX.
12. **CameraX 1.6.1 butuh `compileSdk >= 36`** -- bukan teori, ini kejadian nyata: draft awal
    dokumen ini pakai `compileSdk=35` (dianggap cukup karena di atas `targetSdk`/`minSdk`), tapi
    `assembleDebug` gagal di task `checkDebugAarMetadata` dengan pesan "Dependency
    androidx.camera:... requires ... compile against version 36". Fix: `compileSdk=36` +
    install `platforms;android-36` & `build-tools;36.0.0` lewat sdkmanager. Tidak mempengaruhi
    device compatibility (`targetSdk`/`minSdk` tetap sama). Kalau upgrade versi CameraX lagi di
    masa depan, cek ulang syarat compileSdk-nya di release notes.
13. **`--` di tengah komentar XML itu ilegal** -- draft awal dokumen ini menulis komentar
    `AndroidManifest.xml` dengan format `<!-- A -- B -->` (pakai `--` sebagai pemisah kalimat).
    Ini melanggar spesifikasi XML (comment tidak boleh mengandung `--` di tengah, cuma boleh
    sebagai penutup `-->`) dan bikin `processDebugMainManifest` gagal dengan
    `ManifestMerger2$MergeFailureException` tanpa pesan error yang jelas soal *lokasi* masalahnya.
    Fix: jangan pakai `--` di tengah komentar XML manapun (di manifest maupun layout), pakai `:`
    atau tanda baca lain.

## Verifikasi

1. `.\gradlew.bat tasks` dulu (mengisolasi masalah resolusi dependency sebelum nulis kode app).
2. `.\gradlew.bat assembleDebug` berhasil -> `app\build\outputs\apk\debug\app-debug.apk`.
3. `adb devices` menampilkan Poco M4 Pro sebagai `device` (USB debugging authorized);
   `adb install -r`.
4. Test manual di device: buka app (layar Menu) -> "Buka Kamera" -> kasih izin kamera, lakukan
   satu gerakan isyarat ~1-2 detik, pastikan `<gloss> (confidence%)` muncul dan ter-update (mis.
   "Saya (99,9%)"); pakai `adb logcat *:E` untuk tangkap crash apapun kalau adb tersambung
   (`HandLandmarker.createFromOptions()` gagal, `UnsatisfiedLinkError`, dst -- lihat risiko di
   atas). **Sudah tervalidasi berhasil** di Poco M4 Pro (lihat `PROGRESS.md`) -- tidak ada crash,
   confidence 99,9% pada test pertama.
5. Sanity check handedness: tunjukkan satu tangan yang diketahui, pastikan 63 nilainya masuk ke
   setengah vector yang diharapkan (menangkap masalah mirroring sejak awal).
6. Catatan hasil akhir: label prediksi sekarang ditampilkan sebagai gloss BISINDO (lihat tabel
   mapping di atas, disediakan user -- bukan dari dataset/repo riset ML). Tetap perlu divalidasi
   dengan gerakan yang diketahui artinya untuk cek akurasi mapping-nya benar per index, bukan cuma
   konsistensi index (gerakan sama -> gloss sama berulang kali).
