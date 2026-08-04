package com.example.bisindorecognizer.history

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/** One session's latency for both models -- null mvitLatencyMs means MViTv2 wasn't available. */
data class LatencyEntry(val sessionLabel: String, val cnnLatencyMs: Long, val mvitLatencyMs: Long?)

/**
 * Grouped bar chart: CNNsa-BiLSTM vs MViTv2 inference latency (ms) per session, up to 5 groups
 * (HistoryStore keeps 5 sessions max). Colors are categorical slots 1-2 (blue, orange) from the
 * validated default palette -- fixed order, not cycled, chosen for the two model "identities"
 * (see dataviz skill references/palette.md; this pair clears the CVD-safety gate as an adjacent
 * pair). Direct value labels on each bar instead of a y-axis with ticks -- with at most 5 groups
 * x 2 bars, labeling every bar stays readable without needing tick-formatting logic.
 */
class LatencyChartView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var entries: List<LatencyEntry> = emptyList()

    private val cnnPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#2a78d6") }
    private val mvitPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#eb6834") }
    private val naPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#e1e0d9") }
    private val baselinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#c3c2b7")
        strokeWidth = 3f
    }
    private val valueLabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#0b0b0b")
        textSize = 26f
        textAlign = Paint.Align.CENTER
    }
    private val categoryLabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#52514e")
        textSize = 24f
        textAlign = Paint.Align.CENTER
    }

    fun setEntries(newEntries: List<LatencyEntry>) {
        entries = newEntries
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (entries.isEmpty()) return

        val maxLatency = entries.flatMap { listOfNotNull(it.cnnLatencyMs, it.mvitLatencyMs) }.maxOrNull()?.toFloat() ?: 1f
        val chartTop = 56f
        val baselineY = height - 72f
        val chartHeight = baselineY - chartTop
        val groupWidth = width.toFloat() / entries.size
        val barWidth = groupWidth * 0.28f
        val gap = groupWidth * 0.08f

        entries.forEachIndexed { index, entry ->
            val groupCenterX = groupWidth * index + groupWidth / 2f
            val cnnLeft = groupCenterX - gap / 2f - barWidth
            val mvitLeft = groupCenterX + gap / 2f

            val cnnHeight = (entry.cnnLatencyMs / maxLatency) * chartHeight
            drawBar(canvas, cnnLeft, baselineY, barWidth, cnnHeight, cnnPaint)
            canvas.drawText("${entry.cnnLatencyMs}ms", cnnLeft + barWidth / 2f, baselineY - cnnHeight - 12f, valueLabelPaint)

            if (entry.mvitLatencyMs != null) {
                val mvitHeight = (entry.mvitLatencyMs / maxLatency) * chartHeight
                drawBar(canvas, mvitLeft, baselineY, barWidth, mvitHeight, mvitPaint)
                canvas.drawText("${entry.mvitLatencyMs}ms", mvitLeft + barWidth / 2f, baselineY - mvitHeight - 12f, valueLabelPaint)
            } else {
                drawBar(canvas, mvitLeft, baselineY, barWidth, 8f, naPaint)
                canvas.drawText("N/A", mvitLeft + barWidth / 2f, baselineY - 28f, categoryLabelPaint)
            }

            canvas.drawText(entry.sessionLabel, groupCenterX, height - 20f, categoryLabelPaint)
        }

        canvas.drawLine(16f, baselineY, width - 16f, baselineY, baselinePaint)
    }

    private fun drawBar(canvas: Canvas, left: Float, baselineY: Float, barWidth: Float, barHeight: Float, paint: Paint) {
        val top = baselineY - barHeight
        val radius = 8f
        canvas.drawRoundRect(RectF(left, top, left + barWidth, baselineY + radius), radius, radius, paint)
    }
}
