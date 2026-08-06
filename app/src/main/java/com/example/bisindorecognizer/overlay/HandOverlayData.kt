package com.example.bisindorecognizer.overlay

import android.graphics.PointF

/**
 * One detected hand's 21 landmarks, normalized [0,1] against the RAW analysis bitmap (before
 * rotation is applied for display -- see HandOverlayView.toViewSpace; no mirroring anywhere in
 * this pipeline, back camera never needs it). [label] is MediaPipe's handedness classification
 * used as-is -- correct without any adjustment now that the camera is the back one.
 */
data class OverlayHand(val label: String, val points: List<PointF>)

data class HandOverlayData(
    val hands: List<OverlayHand>,
    val imageWidth: Int,
    val imageHeight: Int,
    val rotationDegrees: Int
)
