package com.example.bisindorecognizer.history

import android.content.Context
import android.graphics.Bitmap
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream

/**
 * Persists a small "flipbook" of overlay-baked JPEG frames per successful (non-timed-out)
 * recording, to app-private internal storage (context.filesDir/history/) -- not MediaStore or
 * a public gallery folder, since this is app-internal review/debug data, not user media, and
 * app-private storage needs no runtime permission. Keeps only the [MAX_SESSIONS] most recent
 * sessions; older ones are deleted automatically after each save.
 */
class HistoryStore(context: Context) {
    private val rootDir = File(context.filesDir, "history").apply { mkdirs() }

    fun saveSession(label: String, confidence: Float, frames: List<Bitmap>) {
        if (frames.isEmpty()) return
        val timestampMs = System.currentTimeMillis()
        val sessionDir = File(rootDir, timestampMs.toString()).apply { mkdirs() }

        frames.forEachIndexed { index, bitmap ->
            FileOutputStream(File(sessionDir, "frame_%02d.jpg".format(index))).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 80, out)
            }
        }

        val metadata = JSONObject().apply {
            put("label", label)
            put("confidence", confidence.toDouble())
            put("timestampMs", timestampMs)
            put("frameCount", frames.size)
        }
        File(sessionDir, "metadata.json").writeText(metadata.toString())

        pruneOldSessions()
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
                label = json.getString("label"),
                confidence = json.getDouble("confidence").toFloat(),
                timestampMs = json.getLong("timestampMs"),
                frameCount = json.getInt("frameCount"),
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
