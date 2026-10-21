package com.example.bisindorecognizer.keypoints

import com.example.bisindorecognizer.inference.OnnxPredictor
import kotlin.math.abs

/**
 * Builds 30-frame keypoint windows for CNNsa-BiLSTM from a live recording.
 *
 * In training, a 30-frame sequence covers about 2 seconds of signing at 15 fps. Live capture can be
 * slower and uneven, so windows are built by time rather than by array index: each window has 30
 * target timestamps spaced 1/15 s apart, and each target takes the captured frame nearest in time.
 * Targets outside the captured range stay zero, like the zero padding used in training.
 */
object KeypointSequenceBuilder {
    private const val TARGET_FRAME_INTERVAL_MS = 1000.0 / 15.0 // training frame rate

    /** Number of windows used by [alignToSlidingWindows]. */
    const val DEFAULT_WINDOW_COUNT = 10

    /**
     * Returns [windowCount] windows whose start times are spread evenly over the recording, so the
     * user does not need to start the sign at an exact moment. The caller classifies every window
     * and keeps the most confident one. If the recording is shorter than one window, a single
     * centered window is returned.
     */
    fun alignToSlidingWindows(
        frames: List<Pair<Long, FloatArray>>,
        windowCount: Int = DEFAULT_WINDOW_COUNT
    ): List<FloatArray> {
        require(frames.isNotEmpty()) { "cannot align an empty frame list" }
        val firstMs = frames.first().first
        val lastMs = frames.last().first
        val windowSpanMs = OnnxPredictor.SEQUENCE_LENGTH * TARGET_FRAME_INTERVAL_MS
        val maxStartOffsetMs = (lastMs - firstMs) - windowSpanMs

        if (windowCount <= 1 || maxStartOffsetMs <= 0.0) {
            val centerMs = (firstMs + lastMs) / 2.0
            return listOf(buildWindow(frames, centerMs - windowSpanMs / 2.0))
        }

        val stepMs = maxStartOffsetMs / (windowCount - 1)
        return (0 until windowCount).map { k ->
            buildWindow(frames, firstMs + k * stepMs)
        }
    }

    private fun buildWindow(frames: List<Pair<Long, FloatArray>>, windowStartMs: Double): FloatArray {
        val targetLength = OnnxPredictor.SEQUENCE_LENGTH
        val featureSize = OnnxPredictor.FEATURE_SIZE
        val out = FloatArray(targetLength * featureSize) // zero-initialized, serves as padding
        val firstMs = frames.first().first
        val lastMs = frames.last().first

        for (i in 0 until targetLength) {
            val targetMs = windowStartMs + i * TARGET_FRAME_INTERVAL_MS
            if (targetMs < firstMs || targetMs > lastMs) continue // outside the recording, stays zero

            var nearest = frames[0]
            var bestDelta = abs(frames[0].first - targetMs)
            for (f in frames) {
                val delta = abs(f.first - targetMs)
                if (delta < bestDelta) {
                    bestDelta = delta
                    nearest = f
                }
            }
            System.arraycopy(nearest.second, 0, out, i * featureSize, featureSize)
        }
        return out
    }
}
