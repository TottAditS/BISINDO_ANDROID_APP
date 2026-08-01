package com.example.bisindorecognizer.keypoints

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log
import com.example.bisindorecognizer.overlay.HandOverlayData
import com.example.bisindorecognizer.overlay.OverlayHand
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt

/**
 * MediaPipe HandLandmarker (LIVE_STREAM). Builds the 126-float vector expected by the ONNX model:
 * left hand -> [0:63], right hand -> [63:126] (by MediaPipe handedness label), each hand
 * normalized by subtracting the wrist (landmark 0) then dividing by the wrist-to-middle-MCP
 * (landmark 9) distance. A hand that isn't detected stays all-zero.
 *
 * Reports one detection at a time via [onDetectionResult]: the raw bitmap that was analyzed (for
 * history-frame compositing), the 126-float model-input vector, and the raw (un-normalized)
 * 21-point landmarks per hand for HandOverlayView's bounding box/skeleton drawing. All three
 * bundled in one callback (rather than separate ones) so callers don't have to rely on call
 * order to correlate them.
 */
class HandLandmarkExtractor(
    context: Context,
    private val onDetectionResult: (bitmap: Bitmap, vector: FloatArray, overlay: HandOverlayData) -> Unit
) {
    private val handLandmarker: HandLandmarker
    private val busy = AtomicBoolean(false)

    // Set right before each detectAsync() call and read back in onResult() -- safe because
    // submit()'s busy flag guarantees only one detection is in flight at a time.
    @Volatile
    private var lastRotationDegrees: Int = 0
    @Volatile
    private lateinit var lastBitmap: Bitmap

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
            .setErrorListener { e -> Log.e(TAG, "HandLandmarker error", e) }
            .build()
        handLandmarker = HandLandmarker.createFromOptions(context, options)
    }

    /**
     * timestampMs must be strictly increasing across calls -- use SystemClock.uptimeMillis()
     * taken at analysis time, NOT System.currentTimeMillis() (can jump due to clock sync) and NOT
     * the ImageProxy sensor timestamp (different clock base/unit, can violate monotonicity and
     * throw).
     */
    fun submit(bitmap: Bitmap, rotationDegrees: Int, timestampMs: Long) {
        // Drop frame if the previous detectAsync call hasn't finished yet -- STRATEGY_KEEP_ONLY_LATEST
        // on ImageAnalysis only controls what reaches the analyzer, not HandLandmarker's own pace.
        if (!busy.compareAndSet(false, true)) return
        lastRotationDegrees = rotationDegrees
        lastBitmap = bitmap
        handLandmarker.detectAsync(BitmapImageBuilder(bitmap).build(), timestampMs)
    }

    private fun onResult(result: HandLandmarkerResult, input: MPImage) {
        busy.set(false)
        val vector = buildVector126(result)
        val overlay = buildOverlayData(result, input.width, input.height, lastRotationDegrees)
        onDetectionResult(lastBitmap, vector, overlay)
    }

    private fun buildOverlayData(
        result: HandLandmarkerResult,
        imageWidth: Int,
        imageHeight: Int,
        rotationDegrees: Int
    ): HandOverlayData {
        val hands = result.landmarks().indices.map { i ->
            val label = result.handedness()[i][0].categoryName() // "Left" atau "Right"
            val points = result.landmarks()[i].map { PointF(it.x(), it.y()) }
            OverlayHand(label, points)
        }
        return HandOverlayData(hands, imageWidth, imageHeight, rotationDegrees)
    }

    private fun buildVector126(result: HandLandmarkerResult): FloatArray {
        val vector = FloatArray(126)
        for (i in result.landmarks().indices) {
            val label = result.handedness()[i][0].categoryName() // "Left" atau "Right"
            val offset = if (label == "Left") 0 else 63
            val lm = result.landmarks()[i]
            val wrist = lm[0]
            val midMcp = lm[9]
            val dx = midMcp.x() - wrist.x()
            val dy = midMcp.y() - wrist.y()
            val dz = midMcp.z() - wrist.z()
            val scale = sqrt(dx * dx + dy * dy + dz * dz)
            for (j in 0 until 21) {
                var x = lm[j].x() - wrist.x()
                var y = lm[j].y() - wrist.y()
                var z = lm[j].z() - wrist.z()
                if (scale >= 1e-6f) {
                    x /= scale; y /= scale; z /= scale
                }
                vector[offset + j * 3] = x
                vector[offset + j * 3 + 1] = y
                vector[offset + j * 3 + 2] = z
            }
        }
        return vector
    }

    fun close() {
        handLandmarker.close()
    }

    companion object {
        private const val TAG = "HandLandmarkExtractor"
    }
}
