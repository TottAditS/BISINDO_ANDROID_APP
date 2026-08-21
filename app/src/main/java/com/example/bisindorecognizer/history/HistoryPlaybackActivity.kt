package com.example.bisindorecognizer.history

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.bisindorecognizer.R
import com.example.bisindorecognizer.databinding.ActivityHistoryPlaybackBinding
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
        val allSessions = HistoryStore(this).listSessions()
        val session = sessionId?.let { id -> allSessions.find { it.id == id } }

        if (session == null) {
            binding.labelText.text = getString(R.string.history_session_missing)
            binding.btnDownloadSequence.isEnabled = false
            return
        }
        // Nomor kronologis (1 = sesi pertama direkam) -- lihat sessionNumbers() -- dipakai di nama
        // file GIF/JSON supaya sesi gampang dibedakan waktu sudah keluar dari aplikasi.
        val sessionNumber = sessionNumbers(allSessions)[session.id]
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

        binding.btnDownloadSequence.setOnClickListener { downloadSequence(session, sessionNumber) }
    }

    private fun downloadSequence(session: HistorySession, sessionNumber: Int?) {
        binding.btnDownloadSequence.isEnabled = false
        // Hentikan flipbook sementara -- dijalankan lagi (mainHandler.post(advanceFrame)) begitu
        // proses ini selesai, sukses maupun gagal, lihat kedua mainHandler.post{} di bawah.
        mainHandler.removeCallbacks(advanceFrame)
        backgroundExecutor.execute {
            // catch (e: Throwable), BUKAN Exception -- OutOfMemoryError itu Error, bukan Exception,
            // "catch (e: Exception)" tidak menangkapnya sama sekali. startActivity() JUGA dibungkus
            // try/catch-nya SENDIRI di dalam mainHandler.post{} -- exception yang terjadi di situ
            // jalan di thread lain (main thread), NANTI setelah try/catch level luar ini sudah
            // selesai, jadi tidak akan pernah tertangkap kalau cuma dibungkus di level luar saja
            // (fitur "Unduh GIF" yang dulu ada di sini kena masalah ini juga sebelum dihapus
            // 2026-08-21 karena tetap crash terus-menerus di device -- lihat PROGRESS.md).
            try {
                val exported = SequenceExporter.export(applicationContext, session, sessionNumber)
                mainHandler.post {
                    binding.btnDownloadSequence.isEnabled = true
                    mainHandler.post(advanceFrame)
                    try {
                        if (exported == null) {
                            Toast.makeText(this, getString(R.string.sequence_export_failed), Toast.LENGTH_SHORT).show()
                        } else {
                            val shareIntent = SequenceExporter.shareIntent(this, exported)
                            startActivity(Intent.createChooser(shareIntent, getString(R.string.sequence_share_chooser_title)))
                        }
                    } catch (e: Throwable) {
                        Log.e(TAG, "Gagal membuka share chooser sequence", e)
                        val msg = getString(R.string.sequence_export_error, e.message ?: e.toString())
                        Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
                    }
                }
            } catch (e: Throwable) {
                Log.e(TAG, "Gagal ekspor sequence JSON", e)
                mainHandler.post {
                    binding.btnDownloadSequence.isEnabled = true
                    mainHandler.post(advanceFrame)
                    // Pesan beda dari sequence_export_failed (session tanpa data) -- ini kasus
                    // exception BENERAN saat proses copy/share, jadi pesan aslinya ditampilkan
                    // langsung di toast supaya diagnosis tidak perlu adb logcat.
                    val msg = getString(R.string.sequence_export_error, e.message ?: e.toString())
                    Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
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
        private const val TAG = "HistoryPlayback"
    }
}
