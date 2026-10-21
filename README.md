# BISINDO Recognizer (Android)

Android application for recognizing word-level Indonesian Sign Language (BISINDO) from the phone
camera. It runs two models on the same recording and shows their results side by side:

- **CNNsa-BiLSTM**, a lightweight model that reads MediaPipe hand keypoints (15.4 MB as ONNX)
- **MViTv2-S**, a video transformer that reads RGB frames (143 MB as ONNX), used for comparison

Everything runs on the device's CPU with ONNX Runtime; the app has no internet permission. The
user interface is in Indonesian.

The models are trained in the research repository
[CNN-LSTM_BISINDO_Android](https://github.com/TottAditS/CNN-LSTM_BISINDO_Android).

## Features

- **Camera**: live preview with a hand skeleton overlay. "Rekam (5 dtk)" records for 5 seconds,
  then runs CNNsa-BiLSTM on 10 sliding windows and MViTv2 on one 16-frame clip. For each model
  the app shows the predicted sign, confidence, latency, and model size. The target class is
  chosen before recording so that correctness is logged automatically.
- **Statistics**: paginated list of all saved attempts (up to 200) with predictions, latency, CPU
  usage, memory, and a correctness toggle per model; per-class charts of latency, CPU, memory, and
  accuracy; export of all attempts to one CSV file.
- **Playback**: flipbook of 10 saved frames per attempt with the overlay, and export of the
  CNNsa-BiLSTM input sequence (`sequence.json`, 30 x 126 values).

## How recognition works

1. CameraX delivers 640x480 frames from the back camera. MediaPipe Hands (Tasks API, LIVE_STREAM)
   detects up to two hands with 21 landmarks each.
2. Each frame becomes a 126-value vector as in training: one hand in [0:63], the other in
   [63:126], each landmark relative to the wrist and divided by the wrist-to-middle-MCP distance.
   A missing hand is all zeros. Two corrections make the live data match the training data: the
   handedness label from the back camera is mirrored, and landmarks are rotated to the upright
   orientation before normalization (see `HandLandmarkExtractor.kt`).
3. During the 5-second recording, frames are kept at about 15 fps, the training frame rate.
4. CNNsa-BiLSTM: `KeypointSequenceBuilder` builds 10 windows of 30 frames, spread over the
   recording and aligned by timestamp. Each window is classified, and the most confident window
   gives the result, so the user does not have to start the sign at an exact moment.
5. MViTv2: `ClipPreprocessor` samples 16 frames uniformly from the recording, resizes them to
   224x224, and normalizes them with the Kinetics-400 mean and standard deviation.
6. CPU usage (process CPU time over wall-clock time) and memory (PSS of the app) are measured
   around each model call (`PerformanceMetrics.kt`).

## Requirements

- JDK 17
- Android SDK with `platforms;android-36`, `build-tools;36.0.0`, and `platform-tools`
  (Android Studio or the command-line tools)
- A phone with Android 8.0 (API 26) or newer and an arm64-v8a CPU

The Gradle wrapper (Gradle 8.13), Android Gradle Plugin 8.13.0, and Kotlin 2.1.20 are configured
in the project.

| Library | Version |
|---|---|
| CameraX | 1.6.1 |
| MediaPipe Tasks Vision | 0.10.35 |
| ONNX Runtime Android | 1.28.0 |

## Build and install

1. Clone this repository.
2. Copy the MViTv2 model files into `app/src/main/assets/`. They are not in this repository
   because `mvit_v2.onnx.data` is larger than GitHub's 100 MB file limit:
   - `mvit_v2.onnx`
   - `mvit_v2.onnx.data`

   Both files are produced by `Export/ExportModels.ipynb` in the research repository, in
   `Export/Output/onnx/`. `cnnsa_bilstm.onnx` and `hand_landmarker.task` are already included.
   Without the MViTv2 files the app still runs and reports MViTv2 as unavailable.
3. Point Gradle to the Android SDK, either with the `ANDROID_HOME` environment variable or a
   `local.properties` file containing `sdk.dir=<path to the SDK>`.
4. Build and install over USB with debugging enabled:

```bash
./gradlew assembleDebug                                   # Windows: .\gradlew.bat assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

On the first launch of the camera screen, the app copies the MViTv2 files to internal storage,
which takes a few seconds.

## Reproducing the on-device test

1. Open the camera screen and choose the target class with the class button.
2. Press "Rekam (5 dtk)" and perform the sign, or play a WL-BISINDO test clip on a monitor in
   front of the camera. Wait until both results appear.
3. Repeat for every attempt. In the study, each of the 32 classes was recorded 5 times on a
   Xiaomi Poco M4 Pro.
4. Open "Statistik", check the correctness toggles, and press "Unduh Semua Statistik (CSV)" to
   export all attempts. The CSV files from the study are in `AndroidEvaluation/` of the research
   repository.

CSV columns: attempt, targetClass, timestampMs, rawFrames, capturedFrames, then for each model
(cnn, mvit) the label, confidence, latency in ms, CPU percent, memory in KB, and correctness.

## Project structure

```
app/src/main/
  AndroidManifest.xml
  assets/
    cnnsa_bilstm.onnx                 CNNsa-BiLSTM
    hand_landmarker.task              MediaPipe hand landmark model
    mvit_v2.onnx, mvit_v2.onnx.data   MViTv2 (copied manually, see above)
  java/com/example/bisindorecognizer/
    MainActivity.kt                   menu
    CameraActivity.kt                 recording and inference of both models
    LabelMap.kt                       class index to gloss (32 classes)
    camera/AppCameraController.kt     CameraX preview and analysis
    keypoints/HandLandmarkExtractor.kt   MediaPipe and the 126-value vector
    keypoints/KeypointSequenceBuilder.kt sliding 30-frame windows
    keypoints/ClipPreprocessor.kt     16-frame clip for MViTv2
    inference/OnnxPredictor.kt        CNNsa-BiLSTM session
    inference/MvitPredictor.kt        MViTv2 session
    metrics/PerformanceMetrics.kt     CPU and memory measurement
    overlay/                          live overlay and annotated frames
    history/                          saved attempts, statistics screen, charts, exports
  res/                                layouts, strings, theme
```

## Classes

Air, Belajar, Cari, Hari, Ingat, Lagi, Maaf, Makan, Motor, Saya, Terima kasih, Tuli, Apa, Siapa,
Kapan, Di mana, Mengapa, Bagaimana, Merah, Kuning, Hijau, Hitam, Dengar, Berangkat, Datang, Teman,
Keluarga, Rumah, Pagi, Siang, Sore, Malam (indices 0 to 31, as in WL-BISINDO).

## Authors

Totti Adithana Sunarto and Benfano Soewito, Master of Computer Science, BINUS Graduate Program,
Bina Nusantara University, Jakarta, Indonesia.
