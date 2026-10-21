package com.example.bisindorecognizer.overlay

import android.graphics.PointF

/**
 * One detected hand: its MediaPipe handedness label and 21 landmarks, normalized to [0, 1] in the
 * unrotated analysis bitmap. Rotation for display is applied when drawing.
 */
data class OverlayHand(val label: String, val points: List<PointF>)

data class HandOverlayData(
    val hands: List<OverlayHand>,
    val imageWidth: Int,
    val imageHeight: Int,
    val rotationDegrees: Int
)
