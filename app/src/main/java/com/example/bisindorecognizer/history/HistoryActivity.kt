package com.example.bisindorecognizer.history

import android.content.Intent
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.bisindorecognizer.databinding.ActivityHistoryBinding

/** Daftar sesi rekam terakhir (maks 5, lihat HistoryStore) -- tap item untuk buka playback. */
class HistoryActivity : AppCompatActivity() {

    private lateinit var binding: ActivityHistoryBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityHistoryBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnCloseHistory.setOnClickListener { finish() }

        val sessions = HistoryStore(this).listSessions()
        binding.historyList.layoutManager = LinearLayoutManager(this)
        binding.historyList.adapter = HistorySessionAdapter(sessions) { session ->
            startActivity(
                Intent(this, HistoryPlaybackActivity::class.java)
                    .putExtra(HistoryPlaybackActivity.EXTRA_SESSION_ID, session.id)
            )
        }
        binding.emptyText.visibility = if (sessions.isEmpty()) View.VISIBLE else View.GONE
    }
}
