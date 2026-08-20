package com.example.bisindorecognizer.gif

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import androidx.core.content.FileProvider
import java.io.File
import java.io.FileOutputStream

/**
 * Encodes a Riwayat session's already-overlay-baked JPEG playback frames into one animated GIF,
 * for the "Unduh GIF" button (HistoryPlaybackActivity) -- lets the user save/share a demo clip
 * without re-recording on camera. Written to app cache (not MediaStore/public gallery), then
 * exposed to the share target via [FileProvider] so no storage permission is ever requested --
 * same zero-extra-permission approach as the rest of the app (see AndroidManifest.xml, only
 * CAMERA is declared).
 */
object GifExporter {
    private const val FRAME_DELAY_MS = 500 // matches HistoryPlaybackActivity's flipbook pacing
    private const val EXPORT_SUBDIR = "gif_exports" // must match res/xml/file_paths.xml <cache-path>

    /** Returns the written .gif file, or null if [frames] is empty. */
    fun export(context: Context, sessionId: String, frames: List<Bitmap>): File? {
        if (frames.isEmpty()) return null
        val dir = File(context.cacheDir, EXPORT_SUBDIR).apply { mkdirs() }
        val outFile = File(dir, "bisindo_$sessionId.gif")
        FileOutputStream(outFile).use { stream ->
            val encoder = GifEncoder(stream)
            encoder.setDelay(FRAME_DELAY_MS)
            encoder.setRepeat(0)
            encoder.start()
            frames.forEach { encoder.addFrame(it) }
            encoder.finish()
        }
        return outFile
    }

    fun shareIntent(context: Context, file: File): Intent {
        val uri: Uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
        return Intent(Intent.ACTION_SEND).apply {
            type = "image/gif"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
    }
}
