package com.example.bisindorecognizer.history

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import java.io.File

/**
 * Shares a session's sequence.json, the 30x126 input sent to CNNsa-BiLSTM, through the Android
 * share sheet. The file is copied to the cache first so the FileProvider grant covers only that copy.
 */
object SequenceExporter {
    private const val EXPORT_SUBDIR = "sequence_exports" // must match res/xml/file_paths.xml

    /**
     * Returns the exported copy, or null if the session has no sequence.json. [sessionNumber] is
     * used in the file name.
     */
    fun export(context: Context, session: HistorySession, sessionNumber: Int?): File? {
        val source = session.sequenceFile()
        if (!source.exists()) return null
        val dir = File(context.cacheDir, EXPORT_SUBDIR).apply { mkdirs() }
        val label = if (sessionNumber != null) "sesi$sessionNumber" else session.id
        val outFile = File(dir, "bisindo_sequence_$label.json")
        source.copyTo(outFile, overwrite = true)
        return outFile
    }

    fun shareIntent(context: Context, file: File): Intent {
        val uri: Uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
        return Intent(Intent.ACTION_SEND).apply {
            type = "application/json"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
    }
}
