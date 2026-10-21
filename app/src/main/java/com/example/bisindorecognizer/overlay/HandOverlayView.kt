package com.example.bisindorecognizer.overlay

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.util.AttributeSet
import android.view.View

/**
 * Draws a bounding box and the 21-point skeleton of each detected hand over the camera preview.
 *
 * Landmarks are normalized to the unrotated analysis bitmap, while PreviewView shows the stream
 * rotated to the device orientation and center-cropped. toViewSpace() applies the same rotation
 * and scaling so the drawing lines up with the hand. The back camera is not mirrored.
 */
class HandOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var overlayData: HandOverlayData? = null

    private val boxPaintLeft = strokePaint(Color.parseColor("#4CAF50"))
    private val boxPaintRight = strokePaint(Color.parseColor("#FF9800"))
    private val skeletonPaint = strokePaint(Color.parseColor("#00E5FF")).apply { strokeWidth = 4f }
    private val jointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#00E5FF")
        style = Paint.Style.FILL
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 36f
        style = Paint.Style.FILL
    }

    private fun strokePaint(colorInt: Int) = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = colorInt
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    fun update(data: HandOverlayData) {
        overlayData = data
        invalidate()
    }

    fun clear() {
        overlayData = null
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val data = overlayData ?: return

        for (hand in data.hands) {
            val viewPoints = hand.points.map {
                toViewSpace(it, data.imageWidth, data.imageHeight, data.rotationDegrees)
            }
            val boxPaint = if (hand.label == "Left") boxPaintLeft else boxPaintRight

            var minX = Float.MAX_VALUE
            var minY = Float.MAX_VALUE
            var maxX = Float.MIN_VALUE
            var maxY = Float.MIN_VALUE
            for (p in viewPoints) {
                if (p.x < minX) minX = p.x
                if (p.y < minY) minY = p.y
                if (p.x > maxX) maxX = p.x
                if (p.y > maxY) maxY = p.y
            }
            canvas.drawRect(minX, minY, maxX, maxY, boxPaint)
            canvas.drawText(hand.label, minX, (minY - 12f).coerceAtLeast(24f), labelPaint)

            for ((a, b) in OverlayRenderer.HAND_CONNECTIONS) {
                val pa = viewPoints[a]
                val pb = viewPoints[b]
                canvas.drawLine(pa.x, pa.y, pb.x, pb.y, skeletonPaint)
            }
            for (p in viewPoints) {
                canvas.drawCircle(p.x, p.y, 6f, jointPaint)
            }
        }
    }

    private fun toViewSpace(pt: PointF, imgW: Int, imgH: Int, rotationDegrees: Int): PointF {
        val rotated = rotateNormalized(pt, rotationDegrees)
        val (rotW, rotH) = if (rotationDegrees == 90 || rotationDegrees == 270) imgH to imgW else imgW to imgH

        val scale = maxOf(width / rotW.toFloat(), height / rotH.toFloat())
        val offsetX = (width - rotW * scale) / 2f
        val offsetY = (height - rotH * scale) / 2f

        val viewX = rotated.x * rotW * scale + offsetX
        val viewY = rotated.y * rotH * scale + offsetY

        return PointF(viewX, viewY)
    }

    private fun rotateNormalized(pt: PointF, rotationDegrees: Int): PointF =
        OverlayRenderer.rotateNormalized(pt, rotationDegrees)
}
