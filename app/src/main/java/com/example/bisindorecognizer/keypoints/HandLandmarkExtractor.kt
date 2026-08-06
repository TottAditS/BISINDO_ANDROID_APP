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
 * left hand -> [0:63], right hand -> [63:126] (by SMOOTHED handedness label, see
 * [smoothedLabels]), each hand normalized by subtracting the wrist (landmark 0) then dividing by
 * the wrist-to-middle-MCP (landmark 9) distance. A hand that isn't detected stays all-zero.
 *
 * Reports one detection at a time via [onDetectionResult]: the bitmap that was analyzed (for
 * history-frame compositing), the 126-float model-input vector, and the raw (un-normalized)
 * 21-point landmarks per hand for HandOverlayView's bounding box/skeleton drawing. All three
 * bundled in one callback (rather than separate ones) so callers don't have to rely on call
 * order to correlate them.
 *
 * HANDEDNESS (see rencana_fix_handedness.md): MediaPipe's Left/Right `handedness()` assumes
 * selfie-mirrored input. This app uses the BACK camera (AppCameraController), which needs no
 * mirroring at any stage -- so `categoryName()` is the correct label MOST of the time. (An
 * earlier attempt kept the front camera and either mirrored the bitmap pre-MediaPipe -- broke
 * overlay geometry, mirror/rotate don't commute for 90/270deg -- or flipped the label string
 * instead; both were reverted once the camera itself switched to back, which needs neither.)
 *
 * FLICKER: MediaPipe's per-frame classification can still transiently flip for ~1s during fast
 * hand movement (observed on-device) even though it's the same physical hand -- LIVE_STREAM mode
 * gives no persistent hand ID to rely on, and a single wrong frame corrupts that frame's
 * [0:63]/[63:126] bucketing, which can corrupt the whole 30-frame CNNsa-BiLSTM sequence built from
 * it. [smoothedLabels] mitigates this with a lightweight nearest-wrist tracker (matches this
 * frame's hands to last frame's by wrist proximity) plus a majority vote over each tracked hand's
 * last [LABEL_HISTORY_SIZE] raw labels -- a single flickered frame is outvoted by the stable
 * majority, but a real, sustained handedness change (e.g. user switches which hand signs) still
 * comes through once it dominates the window.
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

    // Handedness smoothing state -- touched ONLY from onResult() and the functions it calls
    // (buildOverlayData/buildVector126 no longer touch this directly), which all run on
    // MediaPipe's single internal result-delivery thread, sequentially -- no synchronization
    // needed despite being mutable state shared across calls.
    private val trackedHands = mutableListOf<TrackedHand>()

    private class TrackedHand(var wrist: PointF) {
        val labelHistory = ArrayDeque<String>()
    }

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
        val labels = smoothedLabels(result)
        val vector = buildVector126(result, labels)
        val overlay = buildOverlayData(result, labels, input.width, input.height, lastRotationDegrees)
        onDetectionResult(lastBitmap, vector, overlay)
    }

    // Lihat doc class untuk kenapa ini ada. Dipanggil SEKALI per hasil deteksi (bukan per
    // konsumen) supaya buildOverlayData() dan buildVector126() melihat label yang SAMA persis.
    private fun smoothedLabels(result: HandLandmarkerResult): List<String> {
        val wrists = result.landmarks().map { PointF(it[0].x(), it[0].y()) }
        val rawLabels = result.handedness().map { it[0].categoryName() } // "Left" atau "Right"

        val usedTracked = mutableSetOf<Int>()
        val matched = arrayOfNulls<TrackedHand>(wrists.size)
        for (i in wrists.indices) {
            var bestIdx = -1
            var bestDist = Float.MAX_VALUE
            for (j in trackedHands.indices) {
                if (j in usedTracked) continue
                val d = distance(wrists[i], trackedHands[j].wrist)
                if (d < bestDist) {
                    bestDist = d
                    bestIdx = j
                }
            }
            if (bestIdx != -1 && bestDist < MAX_TRACKING_DISTANCE) {
                matched[i] = trackedHands[bestIdx]
                usedTracked.add(bestIdx)
            }
        }

        val nextTrackedHands = mutableListOf<TrackedHand>()
        val smoothed = wrists.indices.map { i ->
            val tracked = matched[i] ?: TrackedHand(wrists[i])
            tracked.wrist = wrists[i]
            tracked.labelHistory.addLast(rawLabels[i])
            if (tracked.labelHistory.size > LABEL_HISTORY_SIZE) tracked.labelHistory.removeFirst()
            nextTrackedHands.add(tracked)
            tracked.labelHistory.groupingBy { it }.eachCount().maxByOrNull { it.value }!!.key
        }
        trackedHands.clear()
        trackedHands.addAll(nextTrackedHands)
        return smoothed
    }

    private fun distance(a: PointF, b: PointF): Float {
        val dx = a.x - b.x
        val dy = a.y - b.y
        return sqrt(dx * dx + dy * dy)
    }

    private fun buildOverlayData(
        result: HandLandmarkerResult,
        labels: List<String>,
        imageWidth: Int,
        imageHeight: Int,
        rotationDegrees: Int
    ): HandOverlayData {
        val hands = result.landmarks().indices.map { i ->
            val points = result.landmarks()[i].map { PointF(it.x(), it.y()) }
            OverlayHand(labels[i], points)
        }
        return HandOverlayData(hands, imageWidth, imageHeight, rotationDegrees)
    }

    private fun buildVector126(result: HandLandmarkerResult, labels: List<String>): FloatArray {
        val vector = FloatArray(126)
        for (i in result.landmarks().indices) {
            val offset = if (labels[i] == "Left") 0 else 63
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

        // Berapa banyak label mentah terakhir (per tangan yang di-track) dipakai buat majority
        // vote -- makin besar makin tahan flicker, tapi makin lambat merespons pertukaran tangan
        // yang BENERAN terjadi (mis. user ganti tangan yang bergestur). Nilai awal, belum ditest.
        private const val LABEL_HISTORY_SIZE = 5

        // Jarak (koordinat ternormalisasi [0,1]) maksimum antar wrist 2 frame berurutan supaya
        // masih dianggap tangan fisik yang sama -- gerakan tangan lebih jauh dari ini di 1 frame
        // dianggap tangan baru (history mulai dari nol lagi). Nilai awal, belum ditest di device.
        private const val MAX_TRACKING_DISTANCE = 0.25f
    }
}
