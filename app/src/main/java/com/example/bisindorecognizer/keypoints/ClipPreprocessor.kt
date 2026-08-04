package com.example.bisindorecognizer.keypoints

import android.graphics.Bitmap
import com.example.bisindorecognizer.inference.MvitPredictor

/**
 * Turns a variable-length sequence of raw RGB frames (already oriented for display -- see
 * OverlayRenderer.orientForDisplay, NOT the raw sensor-orientation bitmap) into the fixed
 * (1,3,16,224,224) clip MViTv2 expects, replicating export_mvit_v2.py / mvit_dataset.py from the
 * ML research repo EXACTLY:
 * 1. Temporal sample/pad to 16 frames: >16 frames -> uniform linspace(0, T-1, 16) sampling (round
 *    to nearest index); <16 frames -> pad by repeating the LAST frame (NOT zero-padding, unlike
 *    CNNsa-BiLSTM's keypoint padding).
 * 2. Resize to 224x224 if not already that size.
 * 3. Normalize with Kinetics-400 stats (mean=0.45, std=0.225, SAME value on all 3 channels --
 *    NOT standard ImageNet mean/std).
 * 4. Transpose (T,H,W,C) -> (C,T,H,W) and flatten (batch dim is implicit -- MvitPredictor adds it
 *    via the tensor shape, not via an extra array dimension here).
 */
object ClipPreprocessor {
    private const val KINETICS_MEAN = 0.45f
    private const val KINETICS_STD = 0.225f

    fun preprocess(frames: List<Bitmap>): FloatArray {
        require(frames.isNotEmpty()) { "cannot preprocess an empty frame list" }

        val sampled = temporalSampleOrPad(frames, MvitPredictor.CLIP_LENGTH)
        val size = MvitPredictor.FRAME_SIZE
        val resized = sampled.map { frame ->
            if (frame.width == size && frame.height == size) frame
            else Bitmap.createScaledBitmap(frame, size, size, true)
        }

        val clipLength = MvitPredictor.CLIP_LENGTH
        val frameArea = size * size
        val out = FloatArray(3 * clipLength * frameArea)
        val pixels = IntArray(frameArea)

        for (t in 0 until clipLength) {
            resized[t].getPixels(pixels, 0, size, 0, 0, size, size)
            for (i in 0 until frameArea) {
                val pixel = pixels[i]
                val r = ((pixel shr 16) and 0xFF) / 255f
                val g = ((pixel shr 8) and 0xFF) / 255f
                val b = (pixel and 0xFF) / 255f
                out[0 * clipLength * frameArea + t * frameArea + i] = (r - KINETICS_MEAN) / KINETICS_STD
                out[1 * clipLength * frameArea + t * frameArea + i] = (g - KINETICS_MEAN) / KINETICS_STD
                out[2 * clipLength * frameArea + t * frameArea + i] = (b - KINETICS_MEAN) / KINETICS_STD
            }
        }
        return out
    }

    private fun temporalSampleOrPad(frames: List<Bitmap>, targetCount: Int): List<Bitmap> {
        val n = frames.size
        if (n == targetCount) return frames
        if (n > targetCount) {
            // linspace(0, n-1, targetCount), rounded to nearest index -- matches mvit_dataset.py.
            return (0 until targetCount).map { i ->
                val pos = i * (n - 1).toFloat() / (targetCount - 1).toFloat()
                frames[Math.round(pos).coerceIn(0, n - 1)]
            }
        }
        // n < targetCount: pad by repeating the LAST frame (not zero-padding).
        val padded = frames.toMutableList()
        val lastFrame = frames.last()
        while (padded.size < targetCount) padded.add(lastFrame)
        return padded
    }
}
