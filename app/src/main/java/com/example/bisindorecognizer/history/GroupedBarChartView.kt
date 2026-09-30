package com.example.bisindorecognizer.history

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/** One group (e.g. one gesture class) -- up to 2 series (CNN/MViT2), null = no data for that series. */
data class ChartGroup(val label: String, val cnnValue: Float?, val mvitValue: Float?)

/**
 * Grouped bar chart for N groups (unlike LatencyChartView's fixed 5-group direct-label design):
 * Y-axis gridlines + ticks instead of per-bar text, since with 32 groups x 2 series direct labels
 * overlap. X-axis labels drawn vertically to fit many groups in limited width.
 */
class GroupedBarChartView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var groups: List<ChartGroup> = emptyList()
    private var unit: String = ""

    private val cnnPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#2a78d6") }
    private val mvitPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#eb6834") }
    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#E1E0D9"); strokeWidth = 2f
    }
    private val axisLabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#52514E"); textSize = 20f
    }
    private val categoryLabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#52514E"); textSize = 18f
        textAlign = Paint.Align.RIGHT
    }

    fun setData(newGroups: List<ChartGroup>, unit: String) {
        groups = newGroups
        this.unit = unit
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (groups.isEmpty()) return

        val maxVal = groups.flatMap { listOfNotNull(it.cnnValue, it.mvitValue) }.maxOrNull()?.coerceAtLeast(0.01f) ?: 1f
        val leftAxisWidth = 90f
        val bottomLabelHeight = 140f
        val chartTop = 16f
        val baselineY = height - bottomLabelHeight
        val chartHeight = baselineY - chartTop
        val chartLeft = leftAxisWidth
        val chartWidth = width - chartLeft - 16f
        val groupWidth = chartWidth / groups.size
        val barWidth = (groupWidth * 0.35f).coerceAtMost(28f)
        val gap = groupWidth * 0.06f

        // Y-axis gridlines (5 ticks)
        for (i in 0..4) {
            val frac = i / 4f
            val y = baselineY - frac * chartHeight
            canvas.drawLine(chartLeft, y, chartLeft + chartWidth, y, gridPaint)
            canvas.drawText("%.0f".format(maxVal * frac), chartLeft - 8f, y + 6f, categoryLabelPaint)
        }

        groups.forEachIndexed { index, group ->
            val groupCenterX = chartLeft + groupWidth * index + groupWidth / 2f
            val cnnLeft = groupCenterX - gap / 2f - barWidth
            val mvitLeft = groupCenterX + gap / 2f

            group.cnnValue?.let { v ->
                val h = (v / maxVal) * chartHeight
                canvas.drawRoundRect(RectF(cnnLeft, baselineY - h, cnnLeft + barWidth, baselineY + 4f), 4f, 4f, cnnPaint)
            }
            group.mvitValue?.let { v ->
                val h = (v / maxVal) * chartHeight
                canvas.drawRoundRect(RectF(mvitLeft, baselineY - h, mvitLeft + barWidth, baselineY + 4f), 4f, 4f, mvitPaint)
            }

            canvas.save()
            canvas.translate(groupCenterX, baselineY + 16f)
            canvas.rotate(60f)
            canvas.drawText(group.label, 0f, 0f, axisLabelPaint.apply { textAlign = Paint.Align.LEFT })
            canvas.restore()
        }

        canvas.drawLine(chartLeft, baselineY, chartLeft + chartWidth, baselineY, gridPaint)
    }
}
