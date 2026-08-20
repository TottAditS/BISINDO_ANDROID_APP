package com.example.bisindorecognizer.camera

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Wraps CameraX ProcessCameraProvider + ImageAnalysis. Named AppCameraController (not
 * CameraController) to avoid ambiguity with androidx.camera.view.CameraController.
 *
 * Uses the BACK camera (see rencana_fix_handedness.md Opsi A) -- deliberately NOT the front/selfie
 * camera, since the front camera's raw sensor feed produced consistently backwards Left/Right
 * `handedness()` labels vs. what MediaPipe expects.
 *
 * NOTE (2026-08-20): switching to the back camera turned out NOT to make MediaPipe's raw
 * Left/Right output match the desktop training pipeline's labels either -- see the HANDEDNESS
 * SWAP doc in HandLandmarkExtractor.buildVector126, which now applies its own correction on top
 * of whatever this camera produces. The back camera is still the right choice here (its output is
 * at least *consistent*, which is what made the swap detectable and fixable); this comment
 * previously claimed its raw output was already correct, which the empirical data contradicted.
 * Trade-off: no live self-view feedback (can't see the overlay skeleton on your own hand while
 * signing) -- accepted deliberately in favor of correct predictions.
 */
class AppCameraController(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    private val onFrame: (bitmap: Bitmap, rotationDegrees: Int, timestampMs: Long) -> Unit
) {
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    fun start() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.surfaceProvider = previewView.surfaceProvider
            }

            // Resolusi analysis diturunkan ke 640x480 (bukan resolusi kamera penuh) -- MediaPipe
            // HandLandmarker memproses tiap frame lebih cepat di resolusi lebih kecil, menaikkan
            // throughput deteksi di device yang CPU-nya jadi bottleneck (lihat diskusi soal
            // capture "patah-patah"/di bawah target 15fps di device lemah -- PROGRESS.md). Deteksi
            // tangan tidak butuh resolusi tinggi, beda dari kebutuhan preview yang tetap full-res
            // (Preview use case terpisah, tidak terpengaruh perubahan ini).
            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                )
                .build()

            val imageAnalysis = ImageAnalysis.Builder()
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setResolutionSelector(resolutionSelector)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                try {
                    val bitmap = imageProxy.toBitmap()
                    val rotation = imageProxy.imageInfo.rotationDegrees
                    onFrame(bitmap, rotation, SystemClock.uptimeMillis())
                } finally {
                    // WAJIB selalu close, kalau tidak pipeline macet setelah beberapa frame.
                    imageProxy.close()
                }
            }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageAnalysis
            )
        }, ContextCompat.getMainExecutor(context))
    }

    fun shutdown() {
        cameraExecutor.shutdown()
    }
}
