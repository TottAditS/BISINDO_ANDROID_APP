package com.example.bisindorecognizer.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer
import kotlin.math.exp

/** Runs cnnsa_bilstm.onnx from the assets and returns the top-K classes of the "logits" output. */
class OnnxPredictor(context: Context) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(
        context.assets.open("cnnsa_bilstm.onnx").use { it.readBytes() },
        OrtSession.SessionOptions()
    )

    /** @return top-[TOP_K] (labelIndex, confidence) pairs, highest confidence first. */
    fun predict(sequence: FloatArray): List<Pair<Int, Float>> {
        val shape = longArrayOf(1, SEQUENCE_LENGTH.toLong(), FEATURE_SIZE.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(sequence), shape).use { inputTensor ->
            session.run(mapOf("keypoints" to inputTensor), setOf("logits")).use { results ->
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
        const val SEQUENCE_LENGTH = 30
        const val FEATURE_SIZE = 126
        const val TOP_K = 3
    }
}
