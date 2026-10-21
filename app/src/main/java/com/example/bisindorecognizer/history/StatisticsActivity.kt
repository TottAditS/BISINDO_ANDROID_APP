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
 * Statistics screen: a paginated list of all saved attempts and four per-class charts (latency,
 * CPU, memory, and accuracy) comparing both models.
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
        reload() // picks up changes made while another screen was open
    }

    private fun reload() {
        // Oldest first, so attempt numbers follow the recording order
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
            // Also update the in-memory list so the accuracy chart uses the new value right away
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
        // The page-size Spinner can call this before reload() has created the adapter
        // (when there are no sessions yet)
        if (!::adapter.isInitialized) return
        // Page 1 shows the newest attempts
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
                Log.e(TAG, "Failed to export the statistics CSV", e)
                runOnUiThread { Toast.makeText(this, "Gagal ekspor CSV: ${e.message}", Toast.LENGTH_LONG).show() }
            }
        }
    }

    // Latency, CPU, and memory need no ground truth, so they are drawn when the screen opens.
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

    // Accuracy uses the correctness toggles, so it is computed on demand with the Calculate button
    // after the toggles have been reviewed.
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
