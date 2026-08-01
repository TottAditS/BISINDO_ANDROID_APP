package com.example.bisindorecognizer.history

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.appcompat.app.AppCompatActivity
import com.example.bisindorecognizer.R
import com.example.bisindorecognizer.databinding.ActivityHistoryPlaybackBinding

/**
 * "Flipbook" playback -- bukan video asli, cuma gonta-ganti ImageView antar frame JPEG yang
 * sudah disimpan HistoryStore (overlay box/skeleton sudah dipanggang langsung ke tiap frame,
 * lihat OverlayRenderer.renderComposite). FRAME_INTERVAL_MS dipilih supaya total durasi kira-kira
 * sama dengan durasi rekam asli (10 frame representasi ~2 detik gerakan -> 200ms/frame).
 */
class HistoryPlaybackActivity : AppCompatActivity() {

    private lateinit var binding: ActivityHistoryPlaybackBinding
    private val mainHandler = Handler(Looper.getMainLooper())
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
            return
        }

        binding.labelText.text = "${session.label} (${"%.1f".format(session.confidence * 100)}%)"
        frames = (0 until session.frameCount).mapNotNull { index ->
            val file = session.frameFile(index)
            if (file.exists()) BitmapFactory.decodeFile(file.path) else null
        }
        mainHandler.post(advanceFrame)
    }

    override fun onDestroy() {
        super.onDestroy()
        mainHandler.removeCallbacks(advanceFrame)
    }

    companion object {
        const val EXTRA_SESSION_ID = "session_id"
        private const val FRAME_INTERVAL_MS = 200L
    }
}
