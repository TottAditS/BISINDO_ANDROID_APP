package com.example.bisindorecognizer.keypoints

import android.graphics.Bitmap
import com.example.bisindorecognizer.inference.MvitPredictor

/**
 * Turns upright RGB frames (see OverlayRenderer.orientForDisplay) into the (3, 16, 224, 224) clip
 * expected by MViTv2, following mvit_dataset.py in the research repository:
 * 1. Sample 16 frames uniformly (linspace, rounded); if there are fewer, repeat the last frame.
 * 2. Resize to 224x224.
 * 3. Normalize with the Kinetics-400 statistics (mean 0.45, std 0.225 on every channel).
 * 4. Lay out as (C, T, H, W). The batch dimension is added by MvitPredictor.
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
            // linspace(0, n - 1, targetCount), rounded to the nearest index
            return (0 until targetCount).map { i ->
                val pos = i * (n - 1).toFloat() / (targetCount - 1).toFloat()
                frames[Math.round(pos).coerceIn(0, n - 1)]
            }
        }
        val padded = frames.toMutableList()
        val lastFrame = frames.last()
        while (padded.size < targetCount) padded.add(lastFrame)
        return padded
    }
}
