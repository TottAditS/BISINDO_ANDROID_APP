package com.example.bisindorecognizer.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer
import kotlin.math.exp

/** Loads cnnsa_bilstm.onnx from assets and runs softmax+argmax over the "logits" output. */
class OnnxPredictor(context: Context) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(
        context.assets.open("cnnsa_bilstm.onnx").use { it.readBytes() },
        OrtSession.SessionOptions()
    )

    fun predict(sequence: FloatArray): Pair<Int, Float> {
        val shape = longArrayOf(1, 30, 126)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(sequence), shape).use { inputTensor ->
            session.run(mapOf("keypoints" to inputTensor), setOf("logits")).use { results ->
                @Suppress("UNCHECKED_CAST")
                val logits = (results.get("logits").get().value as Array<FloatArray>)[0]
                val probs = softmax(logits)
                val idx = probs.indices.maxByOrNull { probs[it] }!!
                return idx to probs[idx]
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
}
