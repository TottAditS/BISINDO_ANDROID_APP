package com.example.bisindorecognizer.history

import java.io.File

/** One ranked prediction, e.g. "Maaf" at 90% confidence. */
data class RankedPrediction(val label: String, val confidence: Float)

/**
 * One saved recording session and the folder that holds its files.
 *
 * - [mvitPredictions] is empty and the MViTv2 metrics are null when MViTv2 was unavailable, failed,
 *   or received no frames.
 * - [frameCount] is the number of playback JPEGs (at most 10). [rawDetectionCount] counts every
 *   MediaPipe result during the recording; [capturedKeypointCount] counts those kept after the
 *   15 fps throttle and passed to the model.
 * - CPU and memory values come from [com.example.bisindorecognizer.metrics.PerformanceMetrics].
 * - [cnnWindowResults] holds the top-1 result of every sliding window, in window order;
 *   [cnnPredictions] is the top-3 of the winning window, and [cnnLatencyMs] covers all windows.
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
    val cnnCpuPercent: Float,
    val cnnRamKb: Int,
    val mvitCpuPercent: Float?,
    val mvitRamKb: Int?,
    val cnnWindowResults: List<RankedPrediction>,
    val targetClass: String?,
    val cnnCorrectOverride: Boolean?,
    val mvitCorrectOverride: Boolean?,
    val dir: File
) {
    fun frameFile(index: Int): File = File(dir, "frame_%02d.jpg".format(index))

    /** The 30x126 input sequence of the winning CNNsa-BiLSTM window. */
    fun sequenceFile(): File = File(dir, "sequence.json")

    /** Null when the session has no target class and no manual override. */
    fun cnnCorrect(): Boolean? =
        cnnCorrectOverride ?: targetClass?.let { it == cnnPredictions.firstOrNull()?.label }

    fun mvitCorrect(): Boolean? =
        mvitCorrectOverride ?: targetClass?.let { it == mvitPredictions.firstOrNull()?.label }
}

/** Chronological session numbers: the first recorded session is 1, whatever the list order. */
fun sessionNumbers(sessions: List<HistorySession>): Map<String, Int> {
    return sessions.sortedBy { it.timestampMs }
        .mapIndexed { index, session -> session.id to index + 1 }
        .toMap()
}

/** Chronological attempt number within each target class. Sessions without a target class are left out. */
fun attemptNumbers(sessions: List<HistorySession>): Map<String, Int> {
    return sessions.filter { it.targetClass != null }
        .groupBy { it.targetClass }
        .flatMap { (_, group) ->
            group.sortedBy { it.timestampMs }.mapIndexed { index, session -> session.id to index + 1 }
        }
        .toMap()
}

/** "90.0% Maaf, 8.0% Datang, 2.0% Saya", optionally prefixed with "[modelName]: ". */
fun formatPredictionBreakdown(predictions: List<RankedPrediction>, modelName: String? = null): String {
    val parts = predictions.joinToString(", ") { "${"%.1f".format(it.confidence * 100)}% ${it.label}" }
    return if (modelName != null) "$modelName: $parts" else parts
}
