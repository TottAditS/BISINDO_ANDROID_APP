package com.example.bisindorecognizer.keypoints

import com.example.bisindorecognizer.inference.OnnxPredictor
import kotlin.math.abs

/**
 * Fits a variable-length, variable-PACING live recording to OnnxPredictor.SEQUENCE_LENGTH (30)
 * frames, matching what CNNsa-BiLSTM's training sequences represent -- see
 * `FeatureExtraction/Scripts/sequence_builder.py::pad_or_crop_sequence` in the ML research repo
 * (CNN-LSTM_BISINDO_Android): training's 30-frame window always corresponds to ~2 seconds of REAL
 * gesture time (extracted at a consistent ~15fps, `PreProcessing/Scripts/frame_extractor.py`).
 *
 * WHY TIME-BASED, not index-based (previous version of this file just took "30 consecutive array
 * entries from the middle"): CameraX's STRATEGY_KEEP_ONLY_LATEST means MediaPipe only gets called
 * on whatever the newest frame is once it's free -- on a CPU-constrained device, that means large
 * gaps in wall-clock time between consecutive CAPTURED entries (frame 1, then frame ~10 worth of
 * real time later, etc). "30 consecutive entries" on such a device can span far more than 2 real
 * seconds, which is a different gesture pacing than training ever saw -- a plausible contributor
 * to CNNsa-BiLSTM being confidently wrong on live predictions even after the temporal-alignment
 * fix that introduced the previous (index-based) version of this file. Selecting by TIMESTAMP
 * instead keeps the 30-frame window anchored to ~2 real seconds regardless of how sparse/uneven
 * the device's actual capture cadence was.
 *
 * Algorithm: build 30 target timestamps spaced [OnnxPredictor.SEQUENCE_LENGTH] * ~66.7ms apart,
 * centered on the middle of the whole capture (mirrors training's "center crop" intent). For each
 * target timestamp, use the nearest-in-time captured frame (duplicate reuse across nearby target
 * slots is expected and fine on sparse devices -- still better than fabricating data). A target
 * timestamp that falls entirely outside the captured range (very short recordings) is left as
 * zero, i.e. PAD_VALUE, matching training's zero-pad for short sequences.
 */
object KeypointSequenceBuilder {
    private const val TARGET_FRAME_INTERVAL_MS = 1000.0 / 15.0 // sama dengan target_fps training

    fun alignToSequenceLength(frames: List<Pair<Long, FloatArray>>): FloatArray {
        require(frames.isNotEmpty()) { "cannot align an empty frame list" }
        val targetLength = OnnxPredictor.SEQUENCE_LENGTH
        val featureSize = OnnxPredictor.FEATURE_SIZE
        val out = FloatArray(targetLength * featureSize) // zero-initialized -- doubles as PAD_VALUE

        val firstMs = frames.first().first
        val lastMs = frames.last().first
        val centerMs = (firstMs + lastMs) / 2.0
        val windowSpanMs = targetLength * TARGET_FRAME_INTERVAL_MS
        val windowStartMs = centerMs - windowSpanMs / 2.0

        for (i in 0 until targetLength) {
            val targetMs = windowStartMs + i * TARGET_FRAME_INTERVAL_MS
            if (targetMs < firstMs || targetMs > lastMs) continue // di luar rentang capture -> tetap nol (pad)

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
