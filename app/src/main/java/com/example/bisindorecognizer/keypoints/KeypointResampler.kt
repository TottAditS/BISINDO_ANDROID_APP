package com.example.bisindorecognizer.keypoints

import com.example.bisindorecognizer.inference.OnnxPredictor

/**
 * Uniformly resamples a variable-length recording (however many frames MediaPipe actually
 * produced during the fixed-duration recording window) down to the fixed
 * OnnxPredictor.SEQUENCE_LENGTH frames the model was trained on, via nearest-neighbor index
 * mapping (no blending between frames). Recording duration is fixed (RECORDING_DURATION_MS in
 * CameraActivity) rather than frame count, since MediaPipe throughput varies by device and by
 * how much per-frame work is happening on its callback thread -- resampling after the fact is
 * what makes that variability not matter (see PROGRESS.md "waktu habis" bug postmortem).
 */
object KeypointResampler {
    fun resampleAndFlatten(frames: List<FloatArray>): FloatArray {
        require(frames.isNotEmpty()) { "cannot resample an empty frame list" }
        val targetCount = OnnxPredictor.SEQUENCE_LENGTH
        val featureSize = OnnxPredictor.FEATURE_SIZE
        val sourceCount = frames.size
        val out = FloatArray(targetCount * featureSize)
        for (i in 0 until targetCount) {
            val sourceIndex = (i * sourceCount / targetCount).coerceIn(0, sourceCount - 1)
            System.arraycopy(frames[sourceIndex], 0, out, i * featureSize, featureSize)
        }
        return out
    }
}
