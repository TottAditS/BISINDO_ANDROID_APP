package com.example.bisindorecognizer.keypoints

/** Rolling window of the last [windowSize] 126-float vectors -- oldest frame drops as new ones arrive. */
class SequenceBuffer(private val windowSize: Int = 30, private val featureSize: Int = 126) {
    private val buffer = ArrayDeque<FloatArray>(windowSize)

    @Synchronized
    fun push(vector: FloatArray) {
        if (buffer.size == windowSize) buffer.removeFirst()
        buffer.addLast(vector)
    }

    @Synchronized
    fun isFull() = buffer.size == windowSize

    @Synchronized
    fun toFlatFloatArray(): FloatArray {
        val out = FloatArray(windowSize * featureSize)
        buffer.forEachIndexed { i, v -> System.arraycopy(v, 0, out, i * featureSize, featureSize) }
        return out
    }
}
