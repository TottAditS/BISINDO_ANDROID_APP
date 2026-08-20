package com.example.bisindorecognizer.history

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.bisindorecognizer.R
import com.example.bisindorecognizer.databinding.ActivityHistoryPlaybackBinding
import com.example.bisindorecognizer.gif.GifExporter
import java.util.concurrent.Executors

/**
 * "Flipbook" playback -- bukan video asli, cuma gonta-ganti ImageView antar frame JPEG yang
 * sudah disimpan HistoryStore (overlay box/skeleton sudah dipanggang langsung ke tiap frame,
 * lihat OverlayRenderer.renderComposite). FRAME_INTERVAL_MS dipilih supaya total durasi kira-kira
 * sama dengan durasi rekam asli (10 frame representasi 5 detik gerakan -> 500ms/frame).
 */
class HistoryPlaybackActivity : AppCompatActivity() {

    private lateinit var binding: ActivityHistoryPlaybackBinding
    private val mainHandler = Handler(Looper.getMainLooper())
    private val backgroundExecutor = Executors.newSingleThreadExecutor()
    private var frames: List<Bitmap> = emptyList()
    private var frameIndex = 0

    private val advanceFrame = object : Runnable {
        override fun run() {
            if (frames.isEmpty()) return
            binding.frameImage.setImageBitmap(frames[frameIndex])
            frameIndex = (frameIndex + 1) % frames.size
            mainHandler.postDelayed(this, FRAME_INTERVAL_MS)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityHistoryPlaybackBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnClosePlayback.setOnClickListener { finish() }

        val sessionId = intent.getStringExtra(EXTRA_SESSION_ID)
        val session = sessionId?.let { id -> HistoryStore(this).listSessions().find { it.id == id } }

        if (session == null) {
            binding.labelText.text = getString(R.string.history_session_missing)
            binding.btnDownloadGif.isEnabled = false
            binding.btnDownloadSequence.isEnabled = false
            return
        }
        binding.btnDownloadSequence.isEnabled = session.sequenceFile().exists()

        val cnnLine = formatPredictionBreakdown(session.cnnPredictions, "CNNsa-BiLSTM")
        val mvitLine = if (session.mvitPredictions.isEmpty()) {
            getString(R.string.history_mvit_unavailable)
        } else {
            formatPredictionBreakdown(session.mvitPredictions, "MViTv2")
        }
        binding.labelText.text = "$cnnLine\n$mvitLine"
        frames = (0 until session.frameCount).mapNotNull { index ->
            val file = session.frameFile(index)
            if (file.exists()) BitmapFactory.decodeFile(file.path) else null
        }
        mainHandler.post(advanceFrame)

        binding.btnDownloadGif.setOnClickListener { downloadGif(session.id) }
        binding.btnDownloadSequence.setOnClickListener { downloadSequence(session) }
    }

    private fun downloadSequence(session: HistorySession) {
        binding.btnDownloadSequence.isEnabled = false
        backgroundExecutor.execute {
            val exported = SequenceExporter.export(applicationContext, session)
            mainHandler.post {
                binding.btnDownloadSequence.isEnabled = true
                if (exported == null) {
                    Toast.makeText(this, getString(R.string.sequence_export_failed), Toast.LENGTH_SHORT).show()
                } else {
                    val shareIntent = SequenceExporter.shareIntent(this, exported)
                    startActivity(Intent.createChooser(shareIntent, getString(R.string.sequence_share_chooser_title)))
                }
            }
        }
    }

    private fun downloadGif(id: String) {
        if (frames.isEmpty()) {
            Toast.makeText(this, getString(R.string.gif_export_failed), Toast.LENGTH_SHORT).show()
            return
        }
        binding.btnDownloadGif.isEnabled = false
        Toast.makeText(this, getString(R.string.gif_export_progress), Toast.LENGTH_SHORT).show()
        backgroundExecutor.execute {
            val gifFile = GifExporter.export(applicationContext, id, frames)
            mainHandler.post {
                binding.btnDownloadGif.isEnabled = true
                if (gifFile == null) {
                    Toast.makeText(this, getString(R.string.gif_export_failed), Toast.LENGTH_SHORT).show()
                } else {
                    val shareIntent = GifExporter.shareIntent(this, gifFile)
                    startActivity(Intent.createChooser(shareIntent, getString(R.string.gif_share_chooser_title)))
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        mainHandler.removeCallbacks(advanceFrame)
        backgroundExecutor.shutdown()
    }

    companion object {
        const val EXTRA_SESSION_ID = "session_id"
        private const val FRAME_INTERVAL_MS = 500L
    }
}
