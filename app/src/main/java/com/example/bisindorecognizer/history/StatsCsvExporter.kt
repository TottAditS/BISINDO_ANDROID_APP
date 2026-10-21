package com.example.bisindorecognizer.history

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import java.io.File

/** Writes all saved sessions, one row per attempt, to a CSV file for analysis. */
object StatsCsvExporter {
    private const val EXPORT_SUBDIR = "stats_exports" // must match res/xml/file_paths.xml
    private val HEADER = listOf(
        "attempt", "targetClass", "timestampMs", "rawFrames", "capturedFrames",
        "cnnLabel", "cnnConfidence", "cnnLatencyMs", "cnnCpuPercent", "cnnRamKb", "cnnCorrect",
        "mvitLabel", "mvitConfidence", "mvitLatencyMs", "mvitCpuPercent", "mvitRamKb", "mvitCorrect"
    )

    fun export(context: Context): File {
        val sessions = HistoryStore(context).listSessions().sortedBy { it.timestampMs }
        val dir = File(context.cacheDir, EXPORT_SUBDIR).apply { mkdirs() }
        val outFile = File(dir, "bisindo_statistik_${System.currentTimeMillis()}.csv")

        outFile.bufferedWriter().use { writer ->
            writer.appendLine(HEADER.joinToString(","))
            sessions.forEachIndexed { index, s ->
                val cnn = s.cnnPredictions.firstOrNull()
                val mvit = s.mvitPredictions.firstOrNull()
                val row = listOf(
                    index + 1, csv(s.targetClass ?: ""), s.timestampMs, s.rawDetectionCount, s.capturedKeypointCount,
                    csv(cnn?.label ?: ""), cnn?.confidence ?: "", s.cnnLatencyMs, s.cnnCpuPercent, s.cnnRamKb, s.cnnCorrect() ?: "",
                    csv(mvit?.label ?: ""), mvit?.confidence ?: "", s.mvitLatencyMs ?: "", s.mvitCpuPercent ?: "", s.mvitRamKb ?: "", s.mvitCorrect() ?: ""
                )
                writer.appendLine(row.joinToString(","))
            }
        }
        return outFile
    }

    private fun csv(s: String): String = "\"${s.replace("\"", "\"\"")}\""

    fun shareIntent(context: Context, file: File): Intent {
        val uri: Uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
        return Intent(Intent.ACTION_SEND).apply {
            type = "text/csv"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
    }
}
