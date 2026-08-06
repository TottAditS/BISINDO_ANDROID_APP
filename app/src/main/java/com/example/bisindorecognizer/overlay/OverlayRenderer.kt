package com.example.bisindorecognizer.overlay

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PointF

/**
 * Geometry + drawing shared by two consumers:
 * - HandOverlayView: live, drawn every frame on top of PreviewView at arbitrary view size with
 *   center-crop scaling (keeps its own cached Paint fields for that hot path -- see that class).
 * - History frame compositing (renderComposite below): infrequent (~10x per 2s recording), bakes
 *   the box+skeleton directly into a copy of the saved bitmap since there's no separate view to
 *   draw on top of at playback time.
 *
 * Both need the SAME rotate-normalized-landmark math so a fix to one (once verified on real
 * hardware -- see HandOverlayView's class doc) applies to the other too.
 */
object OverlayRenderer {

    // Standard 21-point MediaPipe hand connection topology.
    val HAND_CONNECTIONS = listOf(
        0 to 1, 1 to 2, 2 to 3, 3 to 4,
        0 to 5, 5 to 6, 6 to 7, 7 to 8,
        5 to 9, 9 to 10, 10 to 11, 11 to 12,
        9 to 13, 13 to 14, 14 to 15, 15 to 16,
        13 to 17, 17 to 18, 18 to 19, 19 to 20,
        0 to 17
    )

    fun rotateNormalized(pt: PointF, rotationDegrees: Int): PointF {
        return when (((rotationDegrees % 360) + 360) % 360) {
            90 -> PointF(1f - pt.y, pt.x)
            180 -> PointF(1f - pt.x, 1f - pt.y)
            270 -> PointF(pt.y, 1f - pt.x)
            else -> PointF(pt.x, pt.y)
        }
    }

    /**
     * Rotates [source] to match the device's natural orientation. No mirroring -- back camera
     * (AppCameraController) is never mirrored by CameraX/PreviewView, unlike the front camera.
     * No drawing -- used wherever a CLEAN (no box/skeleton) correctly-oriented frame is needed,
     * e.g. MViTv2's input clip, as opposed to renderComposite() below which draws on top of this.
     */
    fun orientForDisplay(source: Bitmap, rotationDegrees: Int): Bitmap {
        val normalizedRotation = ((rotationDegrees % 360) + 360) % 360
        val matrix = Matrix().apply { postRotate(normalizedRotation.toFloat()) }
        val transformed = Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
        return transformed.copy(Bitmap.Config.ARGB_8888, true)
    }

    /** Bakes box+skeleton onto an [orientForDisplay]-ed copy of [source] (does not mutate [source]). */
    fun renderComposite(source: Bitmap, data: HandOverlayData): Bitmap {
        val rotationDegrees = ((data.rotationDegrees % 360) + 360) % 360
        val composed = orientForDisplay(source, rotationDegrees)
        val canvas = Canvas(composed)
        val rotW = composed.width.toFloat()
        val rotH = composed.height.toFloat()

        for (hand in data.hands) {
            val points = hand.points.map { pt ->
                val r = rotateNormalized(pt, rotationDegrees)
                PointF(r.x * rotW, r.y * rotH)
            }
            drawHand(canvas, points, hand.label)
        }
        return composed
    }

    private fun drawHand(canvas: Canvas, points: List<PointF>, label: String) {
        val boxPaint = strokePaint(
            if (label == "Left") Color.parseColor("#4CAF50") else Color.parseColor("#FF9800")
        )
        val skeletonPaint = strokePaint(Color.parseColor("#00E5FF")).apply { strokeWidth = 4f }
        val jointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#00E5FF")
            style = Paint.Style.FILL
        }
        val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 36f
        }

        var minX = Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var maxX = Float.MIN_VALUE
        var maxY = Float.MIN_VALUE
        for (p in points) {
            if (p.x < minX) minX = p.x
            if (p.y < minY) minY = p.y
            if (p.x > maxX) maxX = p.x
            if (p.y > maxY) maxY = p.y
        }
        canvas.drawRect(minX, minY, maxX, maxY, boxPaint)
        canvas.drawText(label, minX, (minY - 12f).coerceAtLeast(24f), labelPaint)

        for ((a, b) in HAND_CONNECTIONS) {
            val pa = points[a]
            val pb = points[b]
            canvas.drawLine(pa.x, pa.y, pb.x, pb.y, skeletonPaint)
        }
        for (p in points) canvas.drawCircle(p.x, p.y, 6f, jointPaint)
    }

    private fun strokePaint(colorInt: Int) = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = colorInt
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }
}
