package com.example.bisindorecognizer.history

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.example.bisindorecognizer.databinding.ItemStatisticsSessionBinding
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Card adapter for one page of attempts; StatisticsActivity handles the pagination. */
class StatisticsSessionAdapter(
    private var sessions: List<HistorySession>,
    private val attemptNumbers: Map<String, Int>,
    private val onLihat: (HistorySession) -> Unit,
    private val onUnduhJson: (HistorySession, Int?) -> Unit,
    private val onHapus: (HistorySession) -> Unit,
    private val onToggleCnn: (HistorySession, Boolean) -> Unit,
    private val onToggleMvit: (HistorySession, Boolean) -> Unit
) : RecyclerView.Adapter<StatisticsSessionAdapter.ViewHolder>() {

    private val dateFormat = SimpleDateFormat("dd MMM yyyy HH:mm:ss", Locale.forLanguageTag("id-ID"))

    inner class ViewHolder(val binding: ItemStatisticsSessionBinding) : RecyclerView.ViewHolder(binding.root)

    fun submitPage(newSessions: List<HistorySession>) {
        sessions = newSessions
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemStatisticsSessionBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val s = sessions[position]
        val b = holder.binding
        val attemptNo = attemptNumbers[s.id]
        val classLabel = if (s.targetClass != null && attemptNo != null) "${s.targetClass} ($attemptNo/5)" else "?"
        b.headerText.text = "${classLabel} -- ${dateFormat.format(Date(s.timestampMs))}"
        b.frameInfoText.text = "Frame: ${s.rawDetectionCount} mentah / ${s.capturedKeypointCount} dipakai"

        val cnnTop = s.cnnPredictions.firstOrNull()
        b.cnnResultText.text = "CNNsa-BiLSTM: ${cnnTop?.label ?: "-"} (${"%.1f".format((cnnTop?.confidence ?: 0f) * 100)}%) -- ${s.cnnLatencyMs} ms"
        b.cnnPerfText.text = "CPU ${"%.0f".format(s.cnnCpuPercent)}% -- RAM ${"%.0f".format(s.cnnRamKb / 1024f)} MB"
        // Clear the listener before setting isChecked, or a recycled view would fire the old listener
        b.cnnCorrectToggle.setOnCheckedChangeListener(null)
        b.cnnCorrectToggle.isChecked = s.cnnCorrect() ?: false
        b.cnnCorrectToggle.setOnCheckedChangeListener { _, checked -> onToggleCnn(s, checked) }

        val mvitTop = s.mvitPredictions.firstOrNull()
        if (mvitTop != null) {
            b.mvitResultText.text = "MViTv2: ${mvitTop.label} (${"%.1f".format(mvitTop.confidence * 100)}%) -- ${s.mvitLatencyMs} ms"
            b.mvitPerfText.text = if (s.mvitCpuPercent != null && s.mvitRamKb != null) {
                "CPU ${"%.0f".format(s.mvitCpuPercent)}% -- RAM ${"%.0f".format(s.mvitRamKb / 1024f)} MB"
            } else ""
        } else {
            b.mvitResultText.text = "MViTv2: -"
            b.mvitPerfText.text = ""
        }
        b.mvitCorrectToggle.setOnCheckedChangeListener(null)
        b.mvitCorrectToggle.isChecked = s.mvitCorrect() ?: false
        b.mvitCorrectToggle.setOnCheckedChangeListener { _, checked -> onToggleMvit(s, checked) }

        val sessNo = attemptNo
        b.btnLihat.setOnClickListener { onLihat(s) }
        b.btnUnduhJson.setOnClickListener { onUnduhJson(s, sessNo) }
        b.btnHapus.setOnClickListener { onHapus(s) }
    }

    override fun getItemCount() = sessions.size
}
