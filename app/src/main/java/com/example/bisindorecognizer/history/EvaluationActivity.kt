package com.example.bisindorecognizer.history

import android.graphics.Typeface
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.bisindorecognizer.R
import com.example.bisindorecognizer.databinding.ActivityEvaluationBinding

/**
 * Ringkasan atas sesi Riwayat yang tersimpan (maks 5, lihat HistoryStore) -- tabel perbandingan
 * CNNsa-BiLSTM vs MViTv2 per sesi, plus grafik batang latency, buat kebutuhan Bab 4 laporan
 * skripsi (trade-off akurasi-vs-efisiensi). Isi tabel per sesi:
 * - Breakdown TOP-3 prediksi tiap model (bukan cuma top-1), format sama dengan Riwayat.
 * - Latency inferensi (ms) tiap model.
 * - Jumlah frame mentah (semua hasil deteksi MediaPipe) vs frame yang benar-benar dipakai model
 *   (setelah throttle ~15fps di CameraActivity) -- diagnostik variasi throughput antar device,
 *   lihat catatan di CameraActivity.nextCaptureFrameElapsedMs.
 * - Ukuran model (statis, sama utk semua sesi).
 *
 * PENTING: ini BUKAN evaluasi akurasi -- tidak ada ground truth (jawaban benar tiap gestur) di
 * layar ini, jadi confidence yang ditampilkan TIDAK SAMA dengan akurasi. Akurasi resmi berasal
 * dari evaluasi test set di desktop (lihat rencana.md / laporan) -- ditulis eksplisit di layar
 * ini (`eval_accuracy_note`) supaya tidak tertukar.
 */
class EvaluationActivity : AppCompatActivity() {

    private lateinit var binding: ActivityEvaluationBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityEvaluationBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnCloseEvaluation.setOnClickListener { finish() }

        // Terlama dulu (#1) supaya urutan tabel & grafik konsisten dengan urutan sesi direkam,
        // beda dari HistoryActivity yang sengaja terbaru dulu untuk browsing cepat.
        val sessions = HistoryStore(this).listSessions().sortedBy { it.timestampMs }
        if (sessions.isEmpty()) {
            binding.emptyText.visibility = View.VISIBLE
            binding.contentGroup.visibility = View.GONE
            return
        }

        buildTable(sessions)
        binding.latencyChart.setEntries(
            sessions.mapIndexed { index, session ->
                LatencyEntry("#${index + 1}", session.cnnLatencyMs, session.mvitLatencyMs)
            }
        )
    }

    // Tiap sel model menampilkan breakdown TOP-3 (bukan cuma top-1) -- format sama persis dengan
    // yang dipakai HistoryPlaybackActivity ("90.0% Maaf, 8.0% Yakin, 2.0% Saya"), lewat fungsi
    // shared formatPredictionBreakdown() supaya konsisten di kedua layar. Kolom Sesi tambah info
    // jumlah frame mentah vs yang dipakai model, buat diagnostik variasi throughput antar device
    // (lihat catatan nextCaptureFrameElapsedMs di CameraActivity).
    private fun buildTable(sessions: List<HistorySession>) {
        addTableRow(
            getString(R.string.eval_col_session), getString(R.string.eval_col_cnn), getString(R.string.eval_col_mvit),
            header = true
        )
        sessions.forEachIndexed { index, session ->
            val sessionText = getString(
                R.string.eval_session_frame_info,
                index + 1, session.rawDetectionCount, session.capturedKeypointCount
            )
            val cnnText = if (session.cnnPredictions.isNotEmpty()) {
                "${formatPredictionBreakdown(session.cnnPredictions)}\n${session.cnnLatencyMs} ms"
            } else {
                "-"
            }
            val mvitText = if (session.mvitPredictions.isNotEmpty() && session.mvitLatencyMs != null) {
                "${formatPredictionBreakdown(session.mvitPredictions)}\n${session.mvitLatencyMs} ms"
            } else {
                getString(R.string.eval_mvit_na)
            }
            addTableRow(sessionText, cnnText, mvitText, header = false)
        }
    }

    private fun addTableRow(col1: String, col2: String, col3: String, header: Boolean) {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
            setPadding(0, 16, 0, 16)
        }
        listOf(col1, col2, col3).forEach { text ->
            row.addView(TextView(this).apply {
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                this.text = text
                gravity = Gravity.CENTER
                textSize = if (header) 14f else 13f
                setTypeface(typeface, if (header) Typeface.BOLD else Typeface.NORMAL)
            })
        }
        binding.tableContainer.addView(row)
    }
}
