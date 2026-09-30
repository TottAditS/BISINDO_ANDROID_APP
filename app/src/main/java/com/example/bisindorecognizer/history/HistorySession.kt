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
 *
 * [cnnCpuPercent]/[cnnRamKb] and [mvitCpuPercent]/[mvitRamKb] (added to close the Ruang Lingkup
 * commitment in Bab 1 -- "penggunaan memori... serta konsumsi CPU" -- previously never measured)
 * are captured by [com.example.bisindorecognizer.metrics.PerformanceMetrics] right around each
 * model's predict() call -- see that object's doc for what they mean and their limitations.
 * MViTv2's are null under the same conditions [mvitPredictions] is empty (unavailable/failed/no
 * frames), not just "measured as zero".
 *
 * [cnnWindowResults] holds the top-1 result of EVERY sliding window tried by
 * [com.example.bisindorecognizer.keypoints.KeypointSequenceBuilder.alignToSlidingWindows] (10 by
 * default, see PROGRESS.md 2026-08-21 "opsi 1"), in window order -- NOT just the winning one that
 * [cnnPredictions] represents. Empty for sessions saved before this field existed. [cnnLatencyMs]
 * for those sessions is the TOTAL time across all windows tried, not one inference.
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

    /** Vector 30x126 mentah yang dikirim ke CNNsa-BiLSTM sesi ini -- lihat HistoryStore.saveSession(). */
    fun sequenceFile(): File = File(dir, "sequence.json")

    /** Null kalau belum ada targetClass (sesi lama) dan belum di-override manual. */
    fun cnnCorrect(): Boolean? =
        cnnCorrectOverride ?: targetClass?.let { it == cnnPredictions.firstOrNull()?.label }

    fun mvitCorrect(): Boolean? =
        mvitCorrectOverride ?: targetClass?.let { it == mvitPredictions.firstOrNull()?.label }
}

/**
 * Memetakan tiap sesi ke nomor urut KRONOLOGIS (sesi paling awal direkam = 1), terlepas dari
 * urutan [sessions] yang diberikan. Dipakai bersama oleh HistoryActivity (daftar Riwayat, sengaja
 * ditampilkan terbaru dulu untuk browsing cepat) dan EvaluationActivity (tabel, sengaja terlama
 * dulu) supaya "Sesi 3" selalu merujuk sesi yang sama persis di kedua layar itu, walau urutan
 * tampilnya beda.
 */
fun sessionNumbers(sessions: List<HistorySession>): Map<String, Int> {
    return sessions.sortedBy { it.timestampMs }
        .mapIndexed { index, session -> session.id to index + 1 }
        .toMap()
}

/** "Percobaan ke-N" per targetClass (1 = pertama direkam untuk kelas itu), kronologis. Sesi
 * tanpa targetClass tidak masuk map ini. */
fun attemptNumbers(sessions: List<HistorySession>): Map<String, Int> {
    return sessions.filter { it.targetClass != null }
        .groupBy { it.targetClass }
        .flatMap { (_, group) ->
            group.sortedBy { it.timestampMs }.mapIndexed { index, session -> session.id to index + 1 }
        }
        .toMap()
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
