package com.example.bisindorecognizer.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.exp

/**
 * Loads mvit_v2.onnx (MViTv2 transformer baseline, for on-device accuracy/latency comparison
 * against OnnxPredictor's CNNsa-BiLSTM). Unlike OnnxPredictor's single small file loaded straight
 * from assets into memory, this model uses ONNX's external-data format (mvit_v2.onnx.data,
 * ~140MB, referenced by mvit_v2.onnx) which ONNX Runtime Android can only load from a real file
 * path -- NOT a byte array -- so both files are copied from assets to context.filesDir on first
 * use (skipped on later launches once already copied there), in the SAME directory since the
 * external-data reference inside the .onnx graph is a relative filename.
 *
 * Output is "logits" ONLY (shape (1,32), no attn_weights) -- different from OnnxPredictor's model
 * which exposes both.
 */
class MvitPredictor(context: Context) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    /** Cold-start/model-load time (copy-if-needed + createSession) -- for comparison table. */
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

    /** @return top-[TOP_K] (labelIndex, confidence) pairs, sorted highest confidence first. */
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
