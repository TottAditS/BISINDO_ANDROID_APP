package com.example.bisindorecognizer

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.bisindorecognizer.camera.AppCameraController
import com.example.bisindorecognizer.databinding.ActivityCameraBinding
import com.example.bisindorecognizer.history.HistoryStore
import com.example.bisindorecognizer.history.RankedPrediction
import com.example.bisindorecognizer.inference.MvitPredictor
import com.example.bisindorecognizer.inference.OnnxPredictor
import com.example.bisindorecognizer.keypoints.ClipPreprocessor
import com.example.bisindorecognizer.keypoints.HandLandmarkExtractor
import com.example.bisindorecognizer.keypoints.KeypointSequenceBuilder
import com.example.bisindorecognizer.metrics.PerformanceMetrics
import com.example.bisindorecognizer.overlay.HandOverlayData
import com.example.bisindorecognizer.overlay.OverlayRenderer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * Camera screen. Each "Record" press captures a fixed-length recording and runs both models on it:
 * CNNsa-BiLSTM on sliding keypoint windows and MViTv2 on one 16-frame RGB clip. The results of both
 * models are shown side by side and saved as one history session.
 *
 * All frames detected during the recording are collected first; the windows and the clip are built
 * afterwards by [KeypointSequenceBuilder] and [ClipPreprocessor].
 */
class CameraActivity : AppCompatActivity() {

    private lateinit var binding: ActivityCameraBinding
    private lateinit var cameraController: AppCameraController
    private lateinit var handLandmarkExtractor: HandLandmarkExtractor
    private lateinit var onnxPredictor: OnnxPredictor
    private lateinit var historyStore: HistoryStore
    private val mainHandler = Handler(Looper.getMainLooper())

    // Loaded once in the background because MViTv2 is large. null means not loaded yet or failed
    // to load; in both cases MViTv2 is skipped and CNNsa-BiLSTM still runs.
    @Volatile
    private var mvitPredictor: MvitPredictor? = null

    // Runs the model loading and all post-recording work off the main thread. A single thread keeps
    // that work serialized.
    private val backgroundExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    // Written from the main thread and read from the MediaPipe result thread.
    private val isRecording = AtomicBoolean(false)
    private var recordingStartElapsedMs = 0L
    private var recordingTargetClass = ""
    private var selectedTargetClass = LabelMap.gloss(0)

    // (elapsed ms since recording start, keypoint vector) for every accepted frame. Guarded by
    // synchronized because the MediaPipe thread writes and the main thread reads.
    private val capturedFrames = mutableListOf<Pair<Long, FloatArray>>()

    // Frames are accepted at about 15 fps, the frame rate used in training, so a 30-frame window
    // covers about the same amount of motion on fast and slow phones.
    private var nextCaptureFrameElapsedMs = 0L

    // Number of MediaPipe results received before throttling, logged for diagnostics.
    private val rawDetectionCount = AtomicInteger(0)

    // One annotated frame is kept every HISTORY_FRAME_INTERVAL_MS for the history playback.
    private var nextHistoryFrameElapsedMs = 0L
    private val pendingHistoryFrames = mutableListOf<Bitmap>()

    // Raw frames and rotations for MViTv2. Rotating and resizing them is deferred to the background
    // thread after recording, because doing it per frame slows down MediaPipe.
    private val mvitRawFrames = mutableListOf<Pair<Bitmap, Int>>()

    companion object {
        private const val TAG = "CameraActivity"
        private const val RECORDING_DURATION_MS = 5000L

        // Same as target_fps in the preprocessing of the research repository.
        private const val TARGET_CAPTURE_FPS = 15L
        private const val TARGET_CAPTURE_FRAME_INTERVAL_MS = 1000L / TARGET_CAPTURE_FPS

        // Same as min_frames in the research preprocessing: shorter videos were dropped from training.
        private const val MIN_KEYPOINT_FRAMES = 10

        private const val HISTORY_FRAMES_PER_SESSION = 10
        private const val HISTORY_FRAME_INTERVAL_MS = RECORDING_DURATION_MS / HISTORY_FRAMES_PER_SESSION
        private const val CNNSA_BILSTM_SIZE_LABEL = "14.65 MB"
        private const val MVIT_V2_SIZE_LABEL = "143 MB"
    }

    private var secondsRemaining = 0
    private val countdownRunnable = object : Runnable {
        override fun run() {
            secondsRemaining--
            if (secondsRemaining <= 0) {
                finishRecordingAndPredict()
            } else {
                binding.predictionText.text = getString(R.string.recording_countdown, secondsRemaining)
                mainHandler.postDelayed(this, 1000L)
            }
        }
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

        // A button with a dialog instead of a Spinner, whose closed state rendered unreadable text
        // on some devices.
        binding.btnTargetClass.text = selectedTargetClass
        binding.btnTargetClass.setOnClickListener {
            val glosses = LabelMap.all().toTypedArray()
            android.app.AlertDialog.Builder(this)
                .setTitle("Pilih kelas target")
                .setItems(glosses) { _, which ->
                    selectedTargetClass = glosses[which]
                    binding.btnTargetClass.text = selectedTargetClass
                }
                .show()
        }

        binding.btnCloseCamera.setOnClickListener { finish() }
        binding.btnRecord.setOnClickListener { startRecording() }

        loadMvitPredictor()

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

    // Recording stays disabled until MViTv2 has either loaded or failed.
    private fun loadMvitPredictor() {
        binding.btnRecord.isEnabled = false
        binding.predictionText.text = getString(R.string.mvit_loading)
        backgroundExecutor.execute {
            try {
                val predictor = MvitPredictor(this)
                mvitPredictor = predictor
                Log.i(TAG, "MViTv2 loaded in ${predictor.loadTimeMs} ms")
                mainHandler.post {
                    binding.btnRecord.isEnabled = true
                    binding.predictionText.text = getString(R.string.prediction_placeholder)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load MViTv2; the comparison is skipped and CNNsa-BiLSTM still runs", e)
                mainHandler.post {
                    binding.btnRecord.isEnabled = true
                    binding.predictionText.text = getString(R.string.mvit_load_failed)
                }
            }
        }
    }

    private fun startRecording() {
        if (!isRecording.compareAndSet(false, true)) return
        synchronized(capturedFrames) {
            capturedFrames.clear()
            nextCaptureFrameElapsedMs = 0L
        }
        rawDetectionCount.set(0)
        synchronized(pendingHistoryFrames) {
            pendingHistoryFrames.clear()
            nextHistoryFrameElapsedMs = 0L
        }
        synchronized(mvitRawFrames) { mvitRawFrames.clear() }
        recordingStartElapsedMs = SystemClock.elapsedRealtime()
        recordingTargetClass = selectedTargetClass

        binding.handOverlay.clear()

        binding.btnRecord.isEnabled = false
        binding.btnRecord.text = getString(R.string.btn_recording)

        secondsRemaining = (RECORDING_DURATION_MS / 1000L).toInt()
        binding.predictionText.text = getString(R.string.recording_countdown, secondsRemaining)
        mainHandler.postDelayed(countdownRunnable, 1000L)
    }

    // Runs on MediaPipe's result thread and must stay cheap: no bitmap rotation or resizing here.
    private fun onDetectionResult(bitmap: Bitmap, vector: FloatArray, overlay: HandOverlayData) {
        // The live skeleton overlay is also shown while recording, so the signer can see what is detected.
        mainHandler.post { binding.handOverlay.update(overlay) }
        if (!isRecording.get()) return
        rawDetectionCount.incrementAndGet()

        val elapsed = SystemClock.elapsedRealtime() - recordingStartElapsedMs

        synchronized(capturedFrames) {
            if (elapsed >= nextCaptureFrameElapsedMs) {
                nextCaptureFrameElapsedMs += TARGET_CAPTURE_FRAME_INTERVAL_MS
                capturedFrames.add(elapsed to vector)
            }
        }

        val shouldSaveHistoryFrame = synchronized(pendingHistoryFrames) {
            if (elapsed >= nextHistoryFrameElapsedMs && pendingHistoryFrames.size < HISTORY_FRAMES_PER_SESSION) {
                nextHistoryFrameElapsedMs += HISTORY_FRAME_INTERVAL_MS
                true
            } else {
                false
            }
        }
        if (shouldSaveHistoryFrame) {
            val composedFrame = OverlayRenderer.renderComposite(bitmap, overlay)
            synchronized(pendingHistoryFrames) { pendingHistoryFrames.add(composedFrame) }
        }

        // Only keeps a reference; no new bitmap is allocated here.
        synchronized(mvitRawFrames) { mvitRawFrames.add(bitmap to overlay.rotationDegrees) }
    }

    // Called when the countdown ends. Snapshots the buffers and runs both models in the background.
    // The record button is re-enabled at the end of every path.
    private fun finishRecordingAndPredict() {
        if (!isRecording.compareAndSet(true, false)) return

        val keypointFrames = synchronized(capturedFrames) { capturedFrames.toList().also { capturedFrames.clear() } }
        val historyFrames = synchronized(pendingHistoryFrames) { pendingHistoryFrames.toList().also { pendingHistoryFrames.clear() } }
        val mvitFrames = synchronized(mvitRawFrames) { mvitRawFrames.toList().also { mvitRawFrames.clear() } }
        val rawFrameCount = rawDetectionCount.get()

        binding.btnRecord.text = getString(R.string.btn_processing)
        binding.predictionText.text = getString(R.string.recording_processing)

        backgroundExecutor.execute {
            if (keypointFrames.size < MIN_KEYPOINT_FRAMES) {
                mainHandler.post {
                    binding.btnRecord.isEnabled = true
                    binding.btnRecord.text = getString(R.string.btn_record)
                    binding.predictionText.text = getString(R.string.recording_no_hand)
                }
                return@execute
            }

            // Sliding-window inference: classify several 30-frame windows spread over the recording
            // and keep the window with the highest top-1 confidence.
            val cnnWindows = KeypointSequenceBuilder.alignToSlidingWindows(keypointFrames)
            val cnnCpuBefore = PerformanceMetrics.snapshotCpu()
            val cnnStart = System.nanoTime()
            val cnnWindowTopK = cnnWindows.map { window -> onnxPredictor.predict(window) }
            val cnnLatencyMs = (System.nanoTime() - cnnStart) / 1_000_000
            val cnnCpuPercent = PerformanceMetrics.cpuUsagePercent(cnnCpuBefore, PerformanceMetrics.snapshotCpu())
            val cnnRamKb = PerformanceMetrics.currentRamKb()

            val bestWindowIndex = cnnWindowTopK.indices.maxBy { cnnWindowTopK[it].first().second }
            val cnnSequence = cnnWindows[bestWindowIndex]
            val cnnTopK = cnnWindowTopK[bestWindowIndex]
            val (topLabelIndex, topConfidence) = cnnTopK.first()
            val cnnResultLine = getString(
                R.string.comparison_row, "CNNsa-BiLSTM", LabelMap.gloss(topLabelIndex), topConfidence * 100,
                cnnLatencyMs, CNNSA_BILSTM_SIZE_LABEL
            )
            val cnnPredictions = cnnTopK.map { (idx, conf) -> RankedPrediction(LabelMap.gloss(idx), conf) }
            // Top-1 of every window, stored for the per-window breakdown in the statistics screen
            val cnnWindowResults = cnnWindowTopK.map { topK ->
                val (idx, conf) = topK.first()
                RankedPrediction(LabelMap.gloss(idx), conf)
            }

            mainHandler.post {
                binding.predictionText.text = "$cnnResultLine\n${getString(R.string.mvit_processing)}"
            }

            val (mvitResultLine, mvitPredictions, mvitLatencyMs, mvitCpuPercent, mvitRamKb) = runMvitComparison(mvitFrames)

            // Saved after MViTv2 finishes so one session holds the results of both models.
            if (historyFrames.isNotEmpty()) {
                historyStore.saveSession(
                    cnnPredictions, mvitPredictions, cnnLatencyMs, mvitLatencyMs, historyFrames,
                    rawFrameCount, keypointFrames.size,
                    cnnCpuPercent, cnnRamKb, mvitCpuPercent, mvitRamKb,
                    cnnSequence, cnnWindowResults, recordingTargetClass
                )
            }

            mainHandler.post {
                binding.btnRecord.isEnabled = true
                binding.btnRecord.text = getString(R.string.btn_record)
                binding.predictionText.text = "$cnnResultLine\n$mvitResultLine"
            }
        }
    }

    private data class MvitComparisonResult(
        val resultLine: String,
        val predictions: List<RankedPrediction>,
        val latencyMs: Long?,
        val cpuPercent: Float?,
        val ramKb: Int?
    )

    // Runs on the background executor. Errors and OOM are reported as text instead of crashing,
    // since a failed MViTv2 run is itself a result worth recording. The metric fields are null when
    // MViTv2 is unavailable, fails, or receives no frames.
    private fun runMvitComparison(frames: List<Pair<Bitmap, Int>>): MvitComparisonResult {
        return try {
            val predictor = mvitPredictor
            when {
                predictor == null -> MvitComparisonResult(getString(R.string.mvit_unavailable), emptyList(), null, null, null)
                frames.isEmpty() -> MvitComparisonResult(getString(R.string.mvit_no_frames), emptyList(), null, null, null)
                else -> {
                    val orientedFrames = frames.map { (bitmap, rotationDegrees) ->
                        val oriented = OverlayRenderer.orientForDisplay(bitmap, rotationDegrees)
                        if (oriented.width == MvitPredictor.FRAME_SIZE && oriented.height == MvitPredictor.FRAME_SIZE) {
                            oriented
                        } else {
                            Bitmap.createScaledBitmap(oriented, MvitPredictor.FRAME_SIZE, MvitPredictor.FRAME_SIZE, true)
                        }
                    }
                    val clip = ClipPreprocessor.preprocess(orientedFrames)
                    val cpuBefore = PerformanceMetrics.snapshotCpu()
                    val start = System.nanoTime()
                    val mvitTopK = predictor.predict(clip)
                    val latencyMs = (System.nanoTime() - start) / 1_000_000
                    val cpuPercent = PerformanceMetrics.cpuUsagePercent(cpuBefore, PerformanceMetrics.snapshotCpu())
                    val ramKb = PerformanceMetrics.currentRamKb()
                    val (topLabelIndex, topConfidence) = mvitTopK.first()
                    val resultLine = getString(
                        R.string.comparison_row, "MViTv2", LabelMap.gloss(topLabelIndex),
                        topConfidence * 100, latencyMs, MVIT_V2_SIZE_LABEL
                    )
                    val predictions = mvitTopK.map { (idx, conf) -> RankedPrediction(LabelMap.gloss(idx), conf) }
                    MvitComparisonResult(resultLine, predictions, latencyMs, cpuPercent, ramKb)
                }
            }
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "MViTv2 ran out of memory during inference", e)
            MvitComparisonResult(getString(R.string.mvit_oom), emptyList(), null, null, null)
        } catch (e: Exception) {
            Log.e(TAG, "MViTv2 inference failed", e)
            MvitComparisonResult(getString(R.string.mvit_error), emptyList(), null, null, null)
        }
    }

    // The camera is released by its lifecycle binding; MediaPipe and the ONNX Runtime sessions are
    // released explicitly. A prediction that is still running when the screen closes is not
    // interrupted; shutdown() only stops new tasks.
    override fun onDestroy() {
        super.onDestroy()
        mainHandler.removeCallbacks(countdownRunnable)
        cameraController.shutdown()
        handLandmarkExtractor.close()
        onnxPredictor.close()
        mvitPredictor?.close()
        backgroundExecutor.shutdown()
    }
}
