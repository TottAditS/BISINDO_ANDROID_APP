package com.example.bisindorecognizer.keypoints

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log
import com.example.bisindorecognizer.overlay.HandOverlayData
import com.example.bisindorecognizer.overlay.OverlayHand
import com.example.bisindorecognizer.overlay.OverlayRenderer
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt

/**
 * MediaPipe HandLandmarker in LIVE_STREAM mode. For every frame it reports the analyzed bitmap, the
 * 126-value model input, and the raw landmarks for the overlay through [onDetectionResult].
 *
 * The model input follows the training layout: one hand in [0:63], the other in [63:126], each
 * expressed relative to the wrist (landmark 0) and divided by the wrist-to-middle-MCP (landmark 9)
 * distance. A missing hand stays zero.
 *
 * Two corrections make the live vectors match the training data, which was checked by replaying
 * recorded sessions through the ONNX model:
 * - The handedness label from the back camera is mirrored relative to the training pipeline, so
 *   "Left" goes to [63:126] and "Right" to [0:63].
 * - MediaPipe receives the unrotated sensor bitmap, so each landmark is rotated to the upright
 *   orientation before normalization, using the frame's rotationDegrees.
 *
 * Detections are also filtered ([plausibleHandIndices]) and handedness labels are smoothed over
 * recent frames ([smoothedLabels]) to reduce misdetections and label flicker.
 */
class HandLandmarkExtractor(
    context: Context,
    private val onDetectionResult: (bitmap: Bitmap, vector: FloatArray, overlay: HandOverlayData) -> Unit
) {
    private val handLandmarker: HandLandmarker
    private val busy = AtomicBoolean(false)

    // Set before each detectAsync() and read in onResult(); safe because only one detection is in flight.
    @Volatile
    private var lastRotationDegrees: Int = 0
    @Volatile
    private lateinit var lastBitmap: Bitmap

    // Only used from MediaPipe's single result thread, so no synchronization is needed.
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
     * timestampMs must increase strictly between calls; use SystemClock.uptimeMillis() taken at
     * analysis time.
     */
    fun submit(bitmap: Bitmap, rotationDegrees: Int, timestampMs: Long) {
        // Drop the frame while the previous detection is still running
        if (!busy.compareAndSet(false, true)) return
        lastRotationDegrees = rotationDegrees
        lastBitmap = bitmap
        handLandmarker.detectAsync(BitmapImageBuilder(bitmap).build(), timestampMs)
    }

    private fun onResult(result: HandLandmarkerResult, input: MPImage) {
        busy.set(false)
        val keptIndices = plausibleHandIndices(result)
        val labels = smoothedLabels(result, keptIndices)
        val vector = buildVector126(result, keptIndices, labels, lastRotationDegrees)
        val overlay = buildOverlayData(result, keptIndices, labels, input.width, input.height, lastRotationDegrees)
        onDetectionResult(lastBitmap, vector, overlay)
    }

    // Drops hands with a low handedness score and keeps only the more confident one when two hands
    // share a label. Dropped hands are treated as not detected in this frame.
    private fun plausibleHandIndices(result: HandLandmarkerResult): List<Int> {
        val confidences = result.handedness().map { it[0].score() }
        val labels = result.handedness().map { it[0].categoryName() }
        val candidates = confidences.indices.filter { confidences[it] >= MIN_HANDEDNESS_CONFIDENCE }

        return candidates
            .groupBy { labels[it] }
            .map { (_, idxs) -> idxs.maxByOrNull { confidences[it] }!! }
    }

    // Matches each hand to the nearest tracked wrist from the previous frame and returns the
    // majority label over its last LABEL_HISTORY_SIZE frames. Keys are indices into result.landmarks().
    private fun smoothedLabels(result: HandLandmarkerResult, keptIndices: List<Int>): Map<Int, String> {
        val wrists = keptIndices.associateWith { PointF(result.landmarks()[it][0].x(), result.landmarks()[it][0].y()) }
        val rawLabels = keptIndices.associateWith { result.handedness()[it][0].categoryName() }

        val usedTracked = mutableSetOf<Int>()
        val matched = keptIndices.associateWith { idx ->
            var bestIdx = -1
            var bestDist = Float.MAX_VALUE
            for (j in trackedHands.indices) {
                if (j in usedTracked) continue
                val d = distance(wrists.getValue(idx), trackedHands[j].wrist)
                if (d < bestDist) {
                    bestDist = d
                    bestIdx = j
                }
            }
            if (bestIdx != -1 && bestDist < MAX_TRACKING_DISTANCE) {
                usedTracked.add(bestIdx)
                trackedHands[bestIdx]
            } else {
                null
            }
        }

        val nextTrackedHands = mutableListOf<TrackedHand>()
        val smoothed = keptIndices.associateWith { idx ->
            val tracked = matched.getValue(idx) ?: TrackedHand(wrists.getValue(idx))
            tracked.wrist = wrists.getValue(idx)
            tracked.labelHistory.addLast(rawLabels.getValue(idx))
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
        keptIndices: List<Int>,
        labels: Map<Int, String>,
        imageWidth: Int,
        imageHeight: Int,
        rotationDegrees: Int
    ): HandOverlayData {
        val hands = keptIndices.map { i ->
            val points = result.landmarks()[i].map { PointF(it.x(), it.y()) }
            OverlayHand(labels.getValue(i), points)
        }
        return HandOverlayData(hands, imageWidth, imageHeight, rotationDegrees)
    }

    // Rotates (x, y) of every landmark to the upright orientation, then normalizes relative to the
    // wrist and the wrist-to-middle-MCP distance, as in the training pipeline.
    private fun buildVector126(
        result: HandLandmarkerResult,
        keptIndices: List<Int>,
        labels: Map<Int, String>,
        rotationDegrees: Int
    ): FloatArray {
        val vector = FloatArray(126)
        for (i in keptIndices) {
            // Mirrored handedness, see the class documentation
            val offset = if (labels.getValue(i) == "Left") 63 else 0
            val lm = result.landmarks()[i]
            val wristRaw = lm[0]
            val wrist = OverlayRenderer.rotateNormalized(PointF(wristRaw.x(), wristRaw.y()), rotationDegrees)
            val midMcpRaw = lm[9]
            val midMcp = OverlayRenderer.rotateNormalized(PointF(midMcpRaw.x(), midMcpRaw.y()), rotationDegrees)
            val dx = midMcp.x - wrist.x
            val dy = midMcp.y - wrist.y
            val dz = midMcpRaw.z() - wristRaw.z()
            val scale = sqrt(dx * dx + dy * dy + dz * dz)
            for (j in 0 until 21) {
                val ptRaw = lm[j]
                val pt = OverlayRenderer.rotateNormalized(PointF(ptRaw.x(), ptRaw.y()), rotationDegrees)
                var x = pt.x - wrist.x
                var y = pt.y - wrist.y
                var z = ptRaw.z() - wristRaw.z()
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

        // Frames used for the handedness majority vote
        private const val LABEL_HISTORY_SIZE = 5

        // Maximum wrist movement between frames, in normalized image coordinates, for a hand to
        // count as the same tracked hand
        private const val MAX_TRACKING_DISTANCE = 0.25f

        // Minimum handedness score; stricter than the 0.5 detection threshold to reject detections
        // on non-hand regions such as a shoulder
        private const val MIN_HANDEDNESS_CONFIDENCE = 0.8f
    }
}
