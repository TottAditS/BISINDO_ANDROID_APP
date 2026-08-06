package com.example.bisindorecognizer.history

import java.io.File

/** One ranked prediction entry -- e.g. "Maaf" at 90% confidence. */
data class RankedPrediction(val label: String, val confidence: Float)

/**
 * One saved recording session: metadata + the folder holding its subsampled JPEG frames.
 * [mvitPredictions] is empty and [mvitLatencyMs] is null when MViTv2 wasn't available for that
 * session (failed to load, OOM, or no frames captured) -- NOT the same as "MViTv2 predicted with
 * low confidence".
 *
 * [frameCount] is the number of JPEG playback thumbnails saved for the Riwayat flipbook (capped
 * at 10, see CameraActivity.HISTORY_FRAMES_PER_SESSION), NOT the number of keypoint frames the
 * model actually used -- that's [capturedKeypointCount]. [rawDetectionCount] is every MediaPipe
 * detection result seen during the recording window BEFORE the ~15fps capture throttle in
 * CameraActivity; [capturedKeypointCount] is how many of those passed the throttle and became
 * input to KeypointSequenceBuilder. Comparing the two tells you the device's raw detection
 * throughput vs. what was actually fed to the model -- see EvaluationActivity.
 */
data class HistorySession(
    val id: String,
    val cnnPredictions: List<RankedPrediction>,
    val mvitPredictions: List<RankedPrediction>,
    val cnnLatencyMs: Long,
    val mvitLatencyMs: Long?,
    val timestampMs: Long,
    val frameCount: Int,
    val rawDetectionCount: Int,
    val capturedKeypointCount: Int,
    val dir: File
) {
    fun frameFile(index: Int): File = File(dir, "frame_%02d.jpg".format(index))
}

/**
 * "90.0% Maaf, 8.0% Yakin, 2.0% Saya" (atau "CNNsa-BiLSTM: 90.0% Maaf, ..." kalau [modelName]
 * diisi) -- top-K predictions dari satu model, paling yakin duluan (list-nya sendiri sudah
 * terurut dari OnnxPredictor/MvitPredictor.predict()). Dipakai bersama oleh
 * HistoryPlaybackActivity (dengan modelName, karena 1 baris = 1 model) dan EvaluationActivity
 * (tanpa modelName, karena nama model sudah jadi header kolom) supaya formatnya konsisten.
 */
fun formatPredictionBreakdown(predictions: List<RankedPrediction>, modelName: String? = null): String {
    val parts = predictions.joinToString(", ") { "${"%.1f".format(it.confidence * 100)}% ${it.label}" }
    return if (modelName != null) "$modelName: $parts" else parts
}
