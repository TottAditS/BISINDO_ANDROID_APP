package com.example.bisindorecognizer.history

import android.app.AlertDialog
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.bisindorecognizer.LabelMap
import com.example.bisindorecognizer.databinding.ActivityStatisticsBinding
import java.util.concurrent.Executors

/**
 * "Statistik" -- daftar berpaginasi semua attempt tersimpan (sampai 200, lihat HistoryStore) +
 * 4 chart agregat per kelas gesture (latensi, CPU, RAM, akurasi). Ganti total dari
 * HistoryActivity + EvaluationActivity lama (5 sesi, tanpa ground truth).
 */
class StatisticsActivity : AppCompatActivity() {

    private lateinit var binding: ActivityStatisticsBinding
    private lateinit var historyStore: HistoryStore
    private val backgroundExecutor = Executors.newSingleThreadExecutor()

    private var allSessions: List<HistorySession> = emptyList()
    private var pageSize = 5
    private var currentPage = 0 // 0-indexed
    private lateinit var adapter: StatisticsSessionAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityStatisticsBinding.inflate(layoutInflater)
        setContentView(binding.root)
        historyStore = HistoryStore(this)

        binding.btnCloseStatistics.setOnClickListener { finish() }

        binding.pageSizeSpinner.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item, listOf("5", "10", "20", "50")
        )
        binding.pageSizeSpinner.setSelection(0)
        binding.pageSizeSpinner.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) {
                pageSize = listOf(5, 10, 20, 50)[position]
                currentPage = 0
                renderPage()
            }
            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
        }

        binding.btnFirst.setOnClickListener { currentPage = 0; renderPage() }
        binding.btnPrev.setOnClickListener { if (currentPage > 0) currentPage--; renderPage() }
        binding.btnNext.setOnClickListener { if (currentPage < lastPageIndex()) currentPage++; renderPage() }
        binding.btnLast.setOnClickListener { currentPage = lastPageIndex(); renderPage() }

        binding.btnDownloadAllStats.setOnClickListener { downloadAllStats() }
        binding.btnCalculate.setOnClickListener { renderAccuracyChart() }

        reload()
    }

    override fun onResume() {
        super.onResume()
        reload() // toggle/hapus di sesi lain (mis. setelah buka HistoryPlaybackActivity) tetap sinkron
    }

    private fun reload() {
        // Terlama dulu supaya nomor attempt kronologis cocok dengan urutan rekam sesungguhnya.
        allSessions = historyStore.listSessions().sortedBy { it.timestampMs }
        if (allSessions.isEmpty()) {
            binding.emptyText.visibility = View.VISIBLE
            binding.contentGroup.visibility = View.GONE
            return
        }
        binding.emptyText.visibility = View.GONE
        binding.contentGroup.visibility = View.VISIBLE

        adapter = StatisticsSessionAdapter(
            emptyList(), attemptNumbers(allSessions),
            onLihat = { s -> startActivity(Intent(this, HistoryPlaybackActivity::class.java).putExtra(HistoryPlaybackActivity.EXTRA_SESSION_ID, s.id)) },
            onUnduhJson = { s, n -> downloadJson(s, n) },
            onHapus = { s -> confirmDelete(s) },
            // Selain tulis ke disk, sinkronkan allSessions in-memory juga -- kalau tidak, chart
            // akurasi (dihitung dari allSessions, dipicu tombol Hitung) masih pakai nilai toggle
            // LAMA sampai activity ini reload() ulang (baru kejadian di onResume), padahal alur
            // yang dimaksud adalah toggle lalu Hitung langsung tanpa keluar-masuk activity dulu.
            onToggleCnn = { s, checked ->
                historyStore.updateCnnCorrectness(s.id, checked)
                allSessions = allSessions.map { if (it.id == s.id) it.copy(cnnCorrectOverride = checked) else it }
            },
            onToggleMvit = { s, checked ->
                historyStore.updateMvitCorrectness(s.id, checked)
                allSessions = allSessions.map { if (it.id == s.id) it.copy(mvitCorrectOverride = checked) else it }
            }
        )
        binding.statisticsList.layoutManager = LinearLayoutManager(this)
        binding.statisticsList.adapter = adapter

        currentPage = currentPage.coerceIn(0, lastPageIndex())
        renderPage()
        renderRateCharts()
    }

    private fun lastPageIndex(): Int = ((allSessions.size - 1) / pageSize).coerceAtLeast(0)

    private fun renderPage() {
        // Guard: kalau belum pernah ada sesi tersimpan, reload() balik lebih awal tanpa
        // menginisialisasi `adapter` -- tapi Spinner (pageSizeSpinner) tetap bisa menembak
        // onItemSelected() secara deferred setelah adapter di-set, terlepas dari visibility
        // parent-nya (contentGroup GONE tidak mencegah callback ini). Tanpa guard ini,
        // renderPage() dipanggil -> crash lateinit property adapter belum diinisialisasi.
        if (!::adapter.isInitialized) return
        // Halaman 1 = attempt TERBARU (terlama dulu di allSessions, jadi ambil dari belakang).
        val newestFirst = allSessions.asReversed()
        val start = (currentPage * pageSize).coerceIn(0, newestFirst.size)
        val end = (start + pageSize).coerceAtMost(newestFirst.size)
        adapter.submitPage(newestFirst.subList(start, end))
        binding.resultCountText.text = "${allSessions.size} Result(s)"
        binding.pageIndicatorText.text = "${currentPage + 1} / ${lastPageIndex() + 1}"
    }

    private fun confirmDelete(session: HistorySession) {
        AlertDialog.Builder(this)
            .setTitle("Hapus attempt?")
            .setMessage("Attempt ini akan dihapus permanen.")
            .setPositiveButton("Hapus") { _, _ ->
                historyStore.deleteSession(session.id)
                reload()
            }
            .setNegativeButton("Batal", null)
            .show()
    }

    private fun downloadJson(session: HistorySession, sessionNumber: Int?) {
        Toast.makeText(this, "Memproses JSON...", Toast.LENGTH_SHORT).show()
        backgroundExecutor.execute {
            try {
                val exported = SequenceExporter.export(this, session, sessionNumber)
                runOnUiThread {
                    if (exported == null) {
                        Toast.makeText(this, "Sesi ini tidak punya data keypoint", Toast.LENGTH_SHORT).show()
                    } else {
                        try {
                            startActivity(Intent.createChooser(SequenceExporter.shareIntent(this, exported), "Bagikan JSON"))
                        } catch (e: Throwable) {
                            Toast.makeText(this, "Gagal berbagi JSON: ${e.message}", Toast.LENGTH_LONG).show()
                        }
                    }
                }
            } catch (e: Throwable) {
                runOnUiThread { Toast.makeText(this, "Gagal ekspor JSON: ${e.message}", Toast.LENGTH_LONG).show() }
            }
        }
    }

    private fun downloadAllStats() {
        Toast.makeText(this, "Memproses CSV...", Toast.LENGTH_SHORT).show()
        backgroundExecutor.execute {
            try {
                val file = StatsCsvExporter.export(this)
                runOnUiThread {
                    try {
                        startActivity(Intent.createChooser(StatsCsvExporter.shareIntent(this, file), "Bagikan CSV Statistik"))
                    } catch (e: Throwable) {
                        Toast.makeText(this, "Gagal berbagi CSV: ${e.message}", Toast.LENGTH_LONG).show()
                    }
                }
            } catch (e: Throwable) {
                Log.e(TAG, "Gagal ekspor CSV statistik", e)
                runOnUiThread { Toast.makeText(this, "Gagal ekspor CSV: ${e.message}", Toast.LENGTH_LONG).show() }
            }
        }
    }

    // Latensi/CPU/RAM tidak butuh ground truth -- bisa langsung dihitung begitu layar dibuka.
    private fun renderRateCharts() {
        val byClass = allSessions.filter { it.targetClass != null }.groupBy { it.targetClass!! }
        val labels = LabelMap.all().filter { byClass.containsKey(it) }

        binding.latencyChart.setData(labels.map { cls ->
            val group = byClass.getValue(cls)
            ChartGroup(cls, group.map { it.cnnLatencyMs.toFloat() }.average().toFloat(),
                group.mapNotNull { it.mvitLatencyMs?.toFloat() }.average().toFloat().takeIf { !it.isNaN() })
        }, "ms")

        binding.cpuChart.setData(labels.map { cls ->
            val group = byClass.getValue(cls)
            ChartGroup(cls, group.map { it.cnnCpuPercent }.average().toFloat(),
                group.mapNotNull { it.mvitCpuPercent }.average().toFloat().takeIf { !it.isNaN() })
        }, "%")

        binding.ramChart.setData(labels.map { cls ->
            val group = byClass.getValue(cls)
            ChartGroup(cls, group.map { it.cnnRamKb / 1024f }.average().toFloat(),
                group.mapNotNull { it.mvitRamKb?.let { r -> r / 1024f } }.average().toFloat().takeIf { !it.isNaN() })
        }, "MB")
    }

    // Akurasi butuh toggle correctness (auto ATAU manual) -- dipicu manual via tombol Hitung
    // supaya user bisa review/override toggle dulu sebelum lihat ringkasannya.
    private fun renderAccuracyChart() {
        val byClass = allSessions.filter { it.targetClass != null }.groupBy { it.targetClass!! }
        val labels = LabelMap.all().filter { byClass.containsKey(it) }
        binding.accuracyChart.setData(labels.map { cls ->
            val group = byClass.getValue(cls)
            val cnnCorrectCount = group.count { it.cnnCorrect() == true }
            val mvitEvaluable = group.filter { it.mvitCorrect() != null }
            val mvitAcc = if (mvitEvaluable.isNotEmpty()) {
                mvitEvaluable.count { it.mvitCorrect() == true } * 100f / mvitEvaluable.size
            } else null
            ChartGroup(cls, cnnCorrectCount * 100f / group.size, mvitAcc)
        }, "%")
    }

    override fun onDestroy() {
        super.onDestroy()
        backgroundExecutor.shutdownNow()
    }

    companion object {
        private const val TAG = "StatisticsActivity"
    }
}
