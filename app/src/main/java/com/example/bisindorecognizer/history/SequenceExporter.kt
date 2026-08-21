package com.example.bisindorecognizer.history

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import java.io.File

/**
 * Shares a session's raw `sequence.json` (the exact 30x126 float array sent to CNNsa-BiLSTM) via
 * the Android share sheet -- e.g. so it can be sent directly in a chat for numeric diagnosis.
 *
 * Copies the file into cache (not the original app-private `files/history/` folder) so the
 * FileProvider grant covers only this one exported copy, not the whole history directory.
 */
object SequenceExporter {
    private const val EXPORT_SUBDIR = "sequence_exports" // must match res/xml/file_paths.xml

    /**
     * Returns the exported copy, or null if [session] predates sequence.json (old session).
     * [sessionNumber] (dari [sessionNumbers], nomor kronologis, 1 = sesi pertama direkam) dipakai
     * di nama file supaya sesi gampang dibedakan waktu dikirim keluar aplikasi (mis. "sesi3") --
     * sebelumnya cuma pakai timestamp mentah, susah dikenali tanpa buka aplikasi lagi.
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
