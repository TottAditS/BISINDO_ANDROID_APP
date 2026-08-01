package com.example.bisindorecognizer.overlay

import android.graphics.PointF

/**
 * One detected hand's 21 landmarks, normalized [0,1] against the RAW analysis bitmap (before
 * rotation/mirroring is applied for display -- see HandOverlayView.toViewSpace).
 */
data class OverlayHand(val label: String, val points: List<PointF>)

data class HandOverlayData(
    val hands: List<OverlayHand>,
    val imageWidth: Int,
    val imageHeight: Int,
    val rotationDegrees: Int
)
