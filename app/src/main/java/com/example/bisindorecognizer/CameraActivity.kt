package com.example.bisindorecognizer

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.bisindorecognizer.camera.AppCameraController
import com.example.bisindorecognizer.databinding.ActivityCameraBinding
import com.example.bisindorecognizer.history.HistoryStore
import com.example.bisindorecognizer.inference.OnnxPredictor
import com.example.bisindorecognizer.keypoints.HandLandmarkExtractor
import com.example.bisindorecognizer.keypoints.SequenceBuffer
import com.example.bisindorecognizer.overlay.HandOverlayData
import com.example.bisindorecognizer.overlay.OverlayRenderer
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Layar kamera + prediksi live. Hidup hanya selama activity ini hidup -- ditutup lewat tombol
 * "Tutup Kamera" atau tombol Back sistem, keduanya berujung ke finish() -> onDestroy(), yang
 * melepas kamera + native resource MediaPipe/ONNX Runtime secara eksplisit (lihat releaseResources()).
 */
class CameraActivity : AppCompatActivity() {

    private lateinit var binding: ActivityCameraBinding
    private lateinit var cameraController: AppCameraController
    private lateinit var handLandmarkExtractor: HandLandmarkExtractor
    private lateinit var onnxPredictor: OnnxPredictor
    private lateinit var historyStore: HistoryStore
    private val sequenceBuffer = SequenceBuffer()
    private val mainHandler = Handler(Looper.getMainLooper())

    // Prediksi HANYA jalan di dalam jendela "Rekam" 2 detik yang dipicu tombol -- bukan sliding
    // window terus-menerus. Model dilatih pakai klip 30-frame yang isinya satu gestur utuh, jadi
    // window acak di tengah gerakan (sliding window lama) menghasilkan prediksi yang tidak berarti.
    // isRecording dibaca/ditulis dari 2 thread (klik tombol di main thread, hasil keypoint di
    // thread internal MediaPipe) makanya AtomicBoolean, dan penyelesaian rekaman (baik karena
    // buffer penuh maupun timeout) di-guard lewat compareAndSet supaya cuma salah satu jalan sekali.
    private val isRecording = AtomicBoolean(false)
    private var recordingTimeoutRunnable: Runnable? = null

    // Untuk histori: dari 30 frame per sesi rekam, cuma simpan 1 dari tiap HISTORY_FRAME_STEP
    // frame (10 total) biar hemat storage -- lihat HistoryStore untuk retensi 5 sesi terakhir.
    // Kedua field ini disentuh dari thread internal MediaPipe (onDetectionResult) DAN main thread
    // (startRecording()/onRecordingTimeout()), makanya akses ke keduanya di-synchronized.
    private var recordingFrameIndex = 0
    private val pendingHistoryFrames = mutableListOf<Bitmap>()

    companion object {
        private const val RECORDING_TIMEOUT_MS = 2000L
        private const val HISTORY_FRAMES_PER_SESSION = 10
        private const val HISTORY_FRAME_STEP = SequenceBuffer.DEFAULT_WINDOW_SIZE / HISTORY_FRAMES_PER_SESSION
    }

    private val requestCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startPipeline() else {
            binding.predictionText.text = getString(R.string.camera_permission_rationale)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        onnxPredictor = OnnxPredictor(this)
        historyStore = HistoryStore(this)
        handLandmarkExtractor = HandLandmarkExtractor(this, ::onDetectionResult)
        cameraController = AppCameraController(
            context = this,
            lifecycleOwner = this,
            previewView = binding.previewView,
            onFrame = { bitmap, rotationDegrees, timestampMs ->
                handLandmarkExtractor.submit(bitmap, rotationDegrees, timestampMs)
            }
        )

        binding.btnCloseCamera.setOnClickListener { finish() }
        binding.btnRecord.setOnClickListener { startRecording() }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startPipeline()
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startPipeline() {
        cameraController.start()
    }

    // Dipanggil dari klik tombol "Rekam" (main thread). Kosongkan buffer supaya jendela 30-frame
    // yang dikumpulkan murni gerakan yang dilakukan SETELAH tombol ditekan, lalu pasang timeout
    // 2 detik sebagai batas atas -- kalau buffer keburu penuh duluan (fps MediaPipe cukup tinggi),
    // completeRecording() lewat onDetectionResult yang menang dan runnable timeout ini dibatalkan.
    private fun startRecording() {
        if (!isRecording.compareAndSet(false, true)) return
        sequenceBuffer.clear()
        synchronized(pendingHistoryFrames) {
            recordingFrameIndex = 0
            pendingHistoryFrames.clear()
        }
        binding.btnRecord.isEnabled = false
        binding.btnRecord.text = getString(R.string.btn_recording)
        binding.predictionText.text = getString(R.string.recording_in_progress)

        val runnable = Runnable { onRecordingTimeout() }
        recordingTimeoutRunnable = runnable
        mainHandler.postDelayed(runnable, RECORDING_TIMEOUT_MS)
    }

    // Callback hasil LIVE_STREAM MediaPipe jalan di thread internal MediaPipe (bukan main thread
    // atau thread executor CameraX). Semua kerja di sini (push buffer, composite bitmap history,
    // predict) cukup ringan untuk jalan langsung di situ, tapi update View WAJIB dipindah ke main
    // thread.
    private fun onDetectionResult(bitmap: Bitmap, vector: FloatArray, overlay: HandOverlayData) {
        mainHandler.post { binding.handOverlay.update(overlay) }

        if (!isRecording.get()) return
        sequenceBuffer.push(vector)

        // pendingHistoryFrames/recordingFrameIndex bisa dibaca/ditulis dari thread ini DAN dari
        // main thread (startRecording()/onRecordingTimeout()) -- synchronized di sini (bukan
        // AtomicBoolean/@Synchronized) karena butuh baca-lalu-tulis 2 field sekaligus.
        val shouldSaveFrame = synchronized(pendingHistoryFrames) {
            val frameIndex = recordingFrameIndex++
            frameIndex % HISTORY_FRAME_STEP == 0 && pendingHistoryFrames.size < HISTORY_FRAMES_PER_SESSION
        }
        if (shouldSaveFrame) {
            val composedFrame = OverlayRenderer.renderComposite(bitmap, overlay)
            synchronized(pendingHistoryFrames) { pendingHistoryFrames.add(composedFrame) }
        }

        if (sequenceBuffer.isFull()) completeRecording()
    }

    private fun completeRecording() {
        if (!isRecording.compareAndSet(true, false)) return
        cancelRecordingTimeout()

        val (labelIndex, confidence) = onnxPredictor.predict(sequenceBuffer.toFlatFloatArray())
        val label = LabelMap.gloss(labelIndex)
        val text = "$label (${"%.1f".format(confidence * 100)}%)"

        val framesToSave = pendingHistoryFrames.toList()
        pendingHistoryFrames.clear()
        historyStore.saveSession(label, confidence, framesToSave)

        mainHandler.post {
            binding.btnRecord.isEnabled = true
            binding.btnRecord.text = getString(R.string.btn_record)
            binding.predictionText.text = text
        }
    }

    // Runnable ini sendiri jalan di main thread (postDelayed di Handler main-looper), jadi aman
    // sentuh View langsung tanpa post lagi.
    private fun onRecordingTimeout() {
        if (!isRecording.compareAndSet(true, false)) return
        recordingTimeoutRunnable = null
        synchronized(pendingHistoryFrames) { pendingHistoryFrames.clear() } // sesi gagal -- tidak masuk histori
        binding.btnRecord.isEnabled = true
        binding.btnRecord.text = getString(R.string.btn_record)
        binding.predictionText.text = getString(R.string.recording_timeout)
    }

    private fun cancelRecordingTimeout() {
        recordingTimeoutRunnable?.let { mainHandler.removeCallbacks(it) }
        recordingTimeoutRunnable = null
    }

    // onDestroy() selalu dipanggil sistem saat activity ini selesai -- baik lewat tombol "Tutup
    // Kamera" (finish() eksplisit), tombol Back, maupun activity di-recreate/dibunuh sistem.
    // ProcessCameraProvider yang di-bind ke lifecycle activity ini (bindToLifecycle di
    // AppCameraController) sudah otomatis melepas kamera saat lifecycle STOPPED/DESTROYED --
    // pemanggilan eksplisit di bawah ini memastikan native resource MediaPipe HandLandmarker dan
    // sesi ONNX Runtime juga dilepas (tidak otomatis diikat ke lifecycle activity), supaya tidak
    // ada apa pun dari layar ini yang tetap jalan di background setelah activity ditutup.
    override fun onDestroy() {
        super.onDestroy()
        cancelRecordingTimeout()
        cameraController.shutdown()
        handLandmarkExtractor.close()
        onnxPredictor.close()
    }
}
