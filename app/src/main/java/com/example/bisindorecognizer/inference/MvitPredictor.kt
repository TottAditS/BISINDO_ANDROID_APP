package com.example.bisindorecognizer.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.exp

/**
 * Runs mvit_v2.onnx, the MViTv2 comparison model.
 *
 * The model stores its weights in an external file, mvit_v2.onnx.data, which ONNX Runtime can only
 * load from a file path. Both files are therefore copied from the assets to filesDir on first use,
 * into the same folder because the graph refers to the weight file by a relative name.
 *
 * The session uses ONNX Runtime's default thread settings. Setting the thread count to all cores
 * made inference slower on the Poco M4 Pro, whose CPU mixes 2 fast and 6 slow cores.
 */
class MvitPredictor(context: Context) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    /** Time to copy the files (first run only) and create the session. */
    val loadTimeMs: Long

    init {
        val modelFile = File(context.filesDir, MODEL_FILE)
        val dataFile = File(context.filesDir, DATA_FILE)

        val start = System.nanoTime()
        if (!modelFile.exists()) copyAsset(context, MODEL_FILE, modelFile)
        if (!dataFile.exists()) copyAsset(context, DATA_FILE, dataFile)
        session = env.createSession(modelFile.absolutePath, OrtSession.SessionOptions())
        loadTimeMs = (System.nanoTime() - start) / 1_000_000
    }

    private fun copyAsset(context: Context, assetName: String, destFile: File) {
        context.assets.open(assetName).use { input ->
            destFile.outputStream().use { output -> input.copyTo(output) }
        }
    }

    /** @return top-[TOP_K] (labelIndex, confidence) pairs, highest confidence first. */
    fun predict(clip: FloatArray): List<Pair<Int, Float>> {
        val shape = longArrayOf(1, 3, CLIP_LENGTH.toLong(), FRAME_SIZE.toLong(), FRAME_SIZE.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(clip), shape).use { inputTensor ->
            session.run(mapOf("video_clip" to inputTensor), setOf("logits")).use { results ->
                @Suppress("UNCHECKED_CAST")
                val logits = (results.get("logits").get().value as Array<FloatArray>)[0]
                val probs = softmax(logits)
                return probs.indices.sortedByDescending { probs[it] }.take(TOP_K).map { it to probs[it] }
            }
        }
    }

    private fun softmax(x: FloatArray): FloatArray {
        val m = x.max()
        val exps = FloatArray(x.size) { exp((x[it] - m).toDouble()).toFloat() }
        val sum = exps.sum()
        return FloatArray(exps.size) { exps[it] / sum }
    }

    fun close() {
        session.close()
    }

    companion object {
        private const val MODEL_FILE = "mvit_v2.onnx"
        private const val DATA_FILE = "mvit_v2.onnx.data"
        const val CLIP_LENGTH = 16
        const val FRAME_SIZE = 224
        const val TOP_K = 3
    }
}
