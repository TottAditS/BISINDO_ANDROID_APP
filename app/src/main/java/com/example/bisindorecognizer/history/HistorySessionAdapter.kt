package com.example.bisindorecognizer.history

import android.graphics.BitmapFactory
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.example.bisindorecognizer.databinding.ItemHistorySessionBinding
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class HistorySessionAdapter(
    private val sessions: List<HistorySession>,
    private val onClick: (HistorySession) -> Unit
) : RecyclerView.Adapter<HistorySessionAdapter.ViewHolder>() {

    private val dateFormat = SimpleDateFormat("dd MMM yyyy HH:mm:ss", Locale.forLanguageTag("id-ID"))

    inner class ViewHolder(val binding: ItemHistorySessionBinding) : RecyclerView.ViewHolder(binding.root)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemHistorySessionBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val session = sessions[position]
        val thumbFile = session.frameFile(0)
        if (thumbFile.exists()) {
            holder.binding.thumbnail.setImageBitmap(BitmapFactory.decodeFile(thumbFile.path))
        }
        val topCnn = session.cnnPredictions.firstOrNull()
        holder.binding.labelText.text = topCnn?.label ?: "?"
        val confidenceText = topCnn?.let { "%.1f".format(it.confidence * 100) } ?: "?"
        holder.binding.detailText.text = "$confidenceText% -- ${dateFormat.format(Date(session.timestampMs))}"
        holder.itemView.setOnClickListener { onClick(session) }
    }

    override fun getItemCount() = sessions.size
}
