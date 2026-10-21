package com.example.bisindorecognizer.history

import android.content.Context
import android.graphics.Bitmap
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream

/**
 * Stores one folder per recording under filesDir/history/ (app-private storage): the playback
 * JPEG frames, the CNNsa-BiLSTM input sequence, and metadata.json. Only the [MAX_SESSIONS] most
 * recent sessions are kept.
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
        cnnSequence: FloatArray,
        cnnWindowResults: List<RankedPrediction>,
        targetClass: String
    ) {
        if (frames.isEmpty() || cnnPredictions.isEmpty()) return
        val timestampMs = System.currentTimeMillis()
        val sessionDir = File(rootDir, timestampMs.toString()).apply { mkdirs() }

        frames.forEachIndexed { index, bitmap ->
            FileOutputStream(File(sessionDir, "frame_%02d.jpg".format(index))).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 80, out)
            }
        }

        // Flat data plus an explicit shape, so numpy.array(data).reshape(shape) restores it
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
            put("cnnWindowResults", toJsonArray(cnnWindowResults))
            put("targetClass", targetClass)
        }
        File(sessionDir, "metadata.json").writeText(metadata.toString())

        pruneOldSessions()
    }

    /**
     * Sets or clears (null) the manual correctness override. Each model has its own method that
     * reads and writes only its own field on disk, so two quick toggles cannot overwrite each other.
     * Does nothing if the session does not exist.
     */
    fun updateCnnCorrectness(sessionId: String, cnnCorrect: Boolean?) {
        val metaFile = File(File(rootDir, sessionId), "metadata.json")
        if (!metaFile.exists()) return
        val json = JSONObject(metaFile.readText())
        if (cnnCorrect != null) json.put("cnnCorrectOverride", cnnCorrect) else json.remove("cnnCorrectOverride")
        metaFile.writeText(json.toString())
    }

    fun updateMvitCorrectness(sessionId: String, mvitCorrect: Boolean?) {
        val metaFile = File(File(rootDir, sessionId), "metadata.json")
        if (!metaFile.exists()) return
        val json = JSONObject(metaFile.readText())
        if (mvitCorrect != null) json.put("mvitCorrectOverride", mvitCorrect) else json.remove("mvitCorrectOverride")
        metaFile.writeText(json.toString())
    }

    fun deleteSession(sessionId: String) {
        File(rootDir, sessionId).deleteRecursively()
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

    // A session with a missing or corrupt metadata.json is skipped instead of breaking the list.
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
                // has() keeps sessions saved by older app versions readable
                rawDetectionCount = if (json.has("rawDetectionCount")) json.getInt("rawDetectionCount") else 0,
                capturedKeypointCount = if (json.has("capturedKeypointCount")) json.getInt("capturedKeypointCount") else 0,
                cnnCpuPercent = if (json.has("cnnCpuPercent")) json.getDouble("cnnCpuPercent").toFloat() else 0f,
                cnnRamKb = if (json.has("cnnRamKb")) json.getInt("cnnRamKb") else 0,
                mvitCpuPercent = if (json.has("mvitCpuPercent")) json.getDouble("mvitCpuPercent").toFloat() else null,
                mvitRamKb = if (json.has("mvitRamKb")) json.getInt("mvitRamKb") else null,
                cnnWindowResults = if (json.has("cnnWindowResults")) fromJsonArray(json.getJSONArray("cnnWindowResults")) else emptyList(),
                targetClass = if (json.has("targetClass")) json.getString("targetClass") else null,
                cnnCorrectOverride = if (json.has("cnnCorrectOverride")) json.getBoolean("cnnCorrectOverride") else null,
                mvitCorrectOverride = if (json.has("mvitCorrectOverride")) json.getBoolean("mvitCorrectOverride") else null,
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
        private const val MAX_SESSIONS = 200
    }
}
