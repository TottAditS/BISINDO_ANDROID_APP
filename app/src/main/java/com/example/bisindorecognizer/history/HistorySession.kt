package com.example.bisindorecognizer.history

import java.io.File

/** One saved recording session: metadata + the folder holding its subsampled JPEG frames. */
data class HistorySession(
    val id: String,
    val label: String,
    val confidence: Float,
    val timestampMs: Long,
    val frameCount: Int,
    val dir: File
) {
    fun frameFile(index: Int): File = File(dir, "frame_%02d.jpg".format(index))
}
