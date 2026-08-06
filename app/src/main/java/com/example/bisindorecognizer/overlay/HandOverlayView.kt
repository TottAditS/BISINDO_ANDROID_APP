package com.example.bisindorecognizer.overlay

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.util.AttributeSet
import android.view.View

/**
 * Draws a bounding box + 21-point skeleton per detected hand on top of the back-camera preview
 * (see AppCameraController/rencana_fix_handedness.md Opsi A for why back, not front/selfie).
 *
 * MediaPipe landmarks are normalized [0,1] against the RAW analysis bitmap (the un-rotated
 * frame HandLandmarker actually ran on). PreviewView shows that same camera stream rotated to the
 * device's natural orientation, scaled with center-crop to fill the view -- NO mirroring, since
 * CameraX/PreviewView never mirrors the back camera (only the front/selfie camera gets that
 * treatment). toViewSpace() replicates the rotation+scale so a box drawn here lands on the same
 * spot as the hand in the preview.
 *
 * Box POSITION confirmed correct on physical hardware with the FRONT camera (2026-08-01,
 * screenshot) -- the rotation/scale math is unchanged after switching to the back camera, only
 * the (now-removed) mirror step differed, so this should still hold, but re-verify on-device
 * after the camera switch.
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
