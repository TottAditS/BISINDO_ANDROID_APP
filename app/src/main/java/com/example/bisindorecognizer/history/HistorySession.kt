package com.example.bisindorecognizer.history

import java.io.File

/** One ranked prediction entry -- e.g. "Maaf" at 90% confidence. */
data class RankedPrediction(val label: String, val confidence: Float)

/**
 * One saved recording session: metadata + the folder holding its subsampled JPEG frames.
 * [mvitPredictions] is empty and [mvitLatencyMs] is null when MViTv2 wasn't available for that
 * session (failed to load, OOM, or no frames captured) -- NOT the same as "MViTv2 predicted with
 * low confidence".
 */
data class HistorySession(
    val id: String,
    val cnnPredictions: List<RankedPrediction>,
    val mvitPredictions: List<RankedPrediction>,
    val cnnLatencyMs: Long,
    val mvitLatencyMs: Long?,
    val timestampMs: Long,
    val frameCount: Int,
    val dir: File
) {
    fun frameFile(index: Int): File = File(dir, "frame_%02d.jpg".format(index))
}
