package com.example.bisindorecognizer.history

import android.content.Context
import android.graphics.Bitmap
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream

/**
 * Persists a small "flipbook" of overlay-baked JPEG frames per successful (non-timed-out)
 * recording, to app-private internal storage (context.filesDir/history/) -- not MediaStore or
 * a public gallery folder, since this is app-internal review/debug data, not user media, and
 * app-private storage needs no runtime permission. Keeps only the [MAX_SESSIONS] most recent
 * sessions; older ones are deleted automatically after each save.
 *
 * Stores top-K predictions from BOTH models (not just the top-1 label) so the history detail
 * screen can show the full confidence breakdown, e.g. "90% Maaf, 8% Yakin, 2% Saya" per model.
 * Also stores per-model inference latency, used by EvaluationActivity's comparison table/chart.
 */
class HistoryStore(context: Context) {
    private val rootDir = File(context.filesDir, "history").apply { mkdirs() }

    fun saveSession(
        cnnPredictions: List<RankedPrediction>,
        mvitPredictions: List<RankedPrediction>,
        cnnLatencyMs: Long,
        mvitLatencyMs: Long?,
        frames: List<Bitmap>,
        rawDetectionCount: Int,
        capturedKeypointCount: Int,
        cnnCpuPercent: Float,
        cnnRamKb: Int,
        mvitCpuPercent: Float?,
        mvitRamKb: Int?,
        cnnSequence: FloatArray
    ) {
        if (frames.isEmpty() || cnnPredictions.isEmpty()) return
        val timestampMs = System.currentTimeMillis()
        val sessionDir = File(rootDir, timestampMs.toString()).apply { mkdirs() }

        frames.forEachIndexed { index, bitmap ->
            FileOutputStream(File(sessionDir, "frame_%02d.jpg".format(index))).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 80, out)
            }
        }

        // Vector 30x126 PERSIS yang dikirim ke CNNsa-BiLSTM (bukan gambar) -- disimpan buat
        // diagnostik: bandingkan angka mentahnya langsung dengan keypoint desktop, dibanding cuma
        // menebak dari screenshot overlay seperti sebelumnya. Format flat array + shape eksplisit
        // (bukan array-of-array) supaya gampang di-load numpy.array(data).reshape(shape) di sisi
        // analisis desktop.
        val sequenceJson = JSONObject().apply {
            put("shape", JSONArray(listOf(30, 126)))
            put("data", JSONArray().apply { cnnSequence.forEach { put(it.toDouble()) } })
        }
        File(sessionDir, "sequence.json").writeText(sequenceJson.toString())

        val metadata = JSONObject().apply {
            put("cnnPredictions", toJsonArray(cnnPredictions))
            put("mvitPredictions", toJsonArray(mvitPredictions))
            put("cnnLatencyMs", cnnLatencyMs)
            if (mvitLatencyMs != null) put("mvitLatencyMs", mvitLatencyMs)
            put("timestampMs", timestampMs)
            put("frameCount", frames.size)
            put("rawDetectionCount", rawDetectionCount)
            put("capturedKeypointCount", capturedKeypointCount)
            put("cnnCpuPercent", cnnCpuPercent.toDouble())
            put("cnnRamKb", cnnRamKb)
            if (mvitCpuPercent != null) put("mvitCpuPercent", mvitCpuPercent.toDouble())
            if (mvitRamKb != null) put("mvitRamKb", mvitRamKb)
        }
        File(sessionDir, "metadata.json").writeText(metadata.toString())

        pruneOldSessions()
    }

    private fun toJsonArray(predictions: List<RankedPrediction>): JSONArray {
        val array = JSONArray()
        for (p in predictions) {
            array.put(JSONObject().apply {
                put("label", p.label)
                put("confidence", p.confidence.toDouble())
            })
        }
        return array
    }

    private fun fromJsonArray(array: JSONArray): List<RankedPrediction> {
        return (0 until array.length()).map { i ->
            val obj = array.getJSONObject(i)
            RankedPrediction(obj.getString("label"), obj.getDouble("confidence").toFloat())
        }
    }

    fun listSessions(): List<HistorySession> {
        val dirs = rootDir.listFiles { file -> file.isDirectory } ?: emptyArray()
        return dirs.mapNotNull(::readMetadata).sortedByDescending { it.timestampMs }
    }

    // Tolerate a missing/corrupt metadata.json (e.g. process killed mid-write) by skipping that
    // session instead of crashing the whole history list.
    private fun readMetadata(dir: File): HistorySession? {
        val metaFile = File(dir, "metadata.json")
        if (!metaFile.exists()) return null
        return try {
            val json = JSONObject(metaFile.readText())
            HistorySession(
                id = dir.name,
                cnnPredictions = fromJsonArray(json.getJSONArray("cnnPredictions")),
                mvitPredictions = fromJsonArray(json.getJSONArray("mvitPredictions")),
                cnnLatencyMs = json.getLong("cnnLatencyMs"),
                mvitLatencyMs = if (json.has("mvitLatencyMs")) json.getLong("mvitLatencyMs") else null,
                timestampMs = json.getLong("timestampMs"),
                frameCount = json.getInt("frameCount"),
                // has() guards: sesi lama (sebelum field ini ditambah) tetap kebaca, cuma tampil 0.
                rawDetectionCount = if (json.has("rawDetectionCount")) json.getInt("rawDetectionCount") else 0,
                capturedKeypointCount = if (json.has("capturedKeypointCount")) json.getInt("capturedKeypointCount") else 0,
                cnnCpuPercent = if (json.has("cnnCpuPercent")) json.getDouble("cnnCpuPercent").toFloat() else 0f,
                cnnRamKb = if (json.has("cnnRamKb")) json.getInt("cnnRamKb") else 0,
                mvitCpuPercent = if (json.has("mvitCpuPercent")) json.getDouble("mvitCpuPercent").toFloat() else null,
                mvitRamKb = if (json.has("mvitRamKb")) json.getInt("mvitRamKb") else null,
                dir = dir
            )
        } catch (e: Exception) {
            null
        }
    }

    private fun pruneOldSessions() {
        listSessions().drop(MAX_SESSIONS).forEach { it.dir.deleteRecursively() }
    }

    companion object {
        private const val MAX_SESSIONS = 5
    }
}
