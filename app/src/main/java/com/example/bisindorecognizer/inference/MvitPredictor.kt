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
 *
 * PERFORMANCE: MViTv2 is a heavy transformer (~34M params, (1,3,16,224,224) input) and was
 * observed taking 18-78 SECONDS per inference on a Poco M4 Pro (CPU-only, no NPU) vs ~400ms on a
 * desktop i7 benchmark. A `setIntraOpNumThreads(availableProcessors())` tweak was tried here and
 * REVERTED -- it made latency WORSE (~200s observed) instead of better, likely because Poco M4
 * Pro's Helio G96 is big.LITTLE (2 fast Cortex-A76 + 6 slow Cortex-A55): forcing ORT to spread
 * intra-op parallelism across all 8 cores means every parallel region waits on the slowest A55
 * thread, which can lose to just letting ORT use its own (apparently more conservative) default.
 * Left at ORT's default (no explicit thread count) -- if revisiting, try a SMALL explicit count
 * (e.g. 2, matching the fast cores) rather than all cores, and measure before keeping it.
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
