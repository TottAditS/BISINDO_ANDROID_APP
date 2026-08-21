package com.example.bisindorecognizer.keypoints

import com.example.bisindorecognizer.inference.OnnxPredictor
import kotlin.math.abs

/**
 * Fits a variable-length, variable-PACING live recording to OnnxPredictor.SEQUENCE_LENGTH (30)
 * frames, matching what CNNsa-BiLSTM's training sequences represent -- see
 * `FeatureExtraction/Scripts/sequence_builder.py::pad_or_crop_sequence` in the ML research repo
 * (CNN-LSTM_BISINDO_Android): training's 30-frame window always corresponds to ~2 seconds of REAL
 * gesture time (extracted at a consistent ~15fps, `PreProcessing/Scripts/frame_extractor.py`).
 *
 * WHY TIME-BASED, not index-based (previous version of this file just took "30 consecutive array
 * entries from the middle"): CameraX's STRATEGY_KEEP_ONLY_LATEST means MediaPipe only gets called
 * on whatever the newest frame is once it's free -- on a CPU-constrained device, that means large
 * gaps in wall-clock time between consecutive CAPTURED entries (frame 1, then frame ~10 worth of
 * real time later, etc). "30 consecutive entries" on such a device can span far more than 2 real
 * seconds, which is a different gesture pacing than training ever saw -- a plausible contributor
 * to CNNsa-BiLSTM being confidently wrong on live predictions even after the temporal-alignment
 * fix that introduced the previous (index-based) version of this file. Selecting by TIMESTAMP
 * instead keeps the 30-frame window anchored to ~2 real seconds regardless of how sparse/uneven
 * the device's actual capture cadence was.
 *
 * Algorithm: build 30 target timestamps spaced [OnnxPredictor.SEQUENCE_LENGTH] * ~66.7ms apart,
 * starting from a given window-start timestamp. For each target timestamp, use the nearest-in-time
 * captured frame (duplicate reuse across nearby target slots is expected and fine on sparse devices
 * -- still better than fabricating data). A target timestamp that falls entirely outside the
 * captured range (very short recordings) is left as zero, i.e. PAD_VALUE, matching training's
 * zero-pad for short sequences.
 *
 * Two entry points: [alignToSequenceLength] builds ONE window centered on the middle of the whole
 * capture (mirrors training's "center crop" intent) -- kept for callers that only need a single
 * sequence. [alignToSlidingWindows] builds several candidate windows spread across the whole
 * recording instead of trusting one centered guess -- see that function's doc for why.
 */
object KeypointSequenceBuilder {
    private const val TARGET_FRAME_INTERVAL_MS = 1000.0 / 15.0 // sama dengan target_fps training

    /** Jumlah jendela geser default untuk [alignToSlidingWindows] -- lihat PROGRESS.md 2026-08-21
     * (opsi 1 "sliding window"). 10x @ ~7ms CNNsa-BiLSTM = ~70-90ms total, jauh di bawah budget
     * real-time (target <100ms per rumusan masalah TETAP terpenuhi meski dijalankan 10x). */
    const val DEFAULT_WINDOW_COUNT = 10

    fun alignToSequenceLength(frames: List<Pair<Long, FloatArray>>): FloatArray {
        require(frames.isNotEmpty()) { "cannot align an empty frame list" }
        val firstMs = frames.first().first
        val lastMs = frames.last().first
        val windowSpanMs = OnnxPredictor.SEQUENCE_LENGTH * TARGET_FRAME_INTERVAL_MS
        val centerMs = (firstMs + lastMs) / 2.0
        return buildWindow(frames, centerMs - windowSpanMs / 2.0)
    }

    /**
     * Ganti pendekatan "satu jendela di tengah waktu rekaman" dengan MENCOBA [windowCount] jendela
     * 30-frame yang posisi mulainya digeser merata di sepanjang seluruh rekaman (bukan cuma di
     * tengah) -- lihat diskusi PROGRESS.md 2026-08-21 "opsi 1 sliding window". Alasannya: analisis
     * numerik terhadap sequence.json run2 menunjukkan yang benar-benar menentukan benar/salahnya
     * prediksi itu SEBERAPA BANYAK frame yang berhasil dapat sinyal tangan di dalam jendela 30-frame
     * itu (coverage), bukan seberapa presisi jendela itu dipusatkan ke tengah rekaman -- sesi yang
     * benar semua punya >=15/30 frame terisi meski posisinya tidak center dan ada celah, sesi yang
     * salah cuma 4/30 terisi. Dengan mencoba banyak posisi, salah satu di antaranya kemungkinan
     * besar akan "menangkap" bagian rekaman yang cakupannya paling baik, tanpa pengguna perlu
     * mengira-ngira detik keberapa harus mulai bergerak.
     *
     * Setiap jendela dibangun dengan logika nearest-neighbor-by-timestamp yang sama persis dengan
     * [alignToSequenceLength], cuma titik awalnya digeser. Kalau rentang rekaman lebih pendek dari
     * satu jendela penuh (tidak ada ruang buat geser), kembalikan 1 jendela saja (tengah), sama
     * seperti sebelumnya -- mengulang jendela identik 10x cuma buang-buang CPU tanpa manfaat.
     */
    fun alignToSlidingWindows(
        frames: List<Pair<Long, FloatArray>>,
        windowCount: Int = DEFAULT_WINDOW_COUNT
    ): List<FloatArray> {
        require(frames.isNotEmpty()) { "cannot align an empty frame list" }
        val firstMs = frames.first().first
        val lastMs = frames.last().first
        val windowSpanMs = OnnxPredictor.SEQUENCE_LENGTH * TARGET_FRAME_INTERVAL_MS
        val maxStartOffsetMs = (lastMs - firstMs) - windowSpanMs

        if (windowCount <= 1 || maxStartOffsetMs <= 0.0) {
            val centerMs = (firstMs + lastMs) / 2.0
            return listOf(buildWindow(frames, centerMs - windowSpanMs / 2.0))
        }

        val stepMs = maxStartOffsetMs / (windowCount - 1)
        return (0 until windowCount).map { k ->
            buildWindow(frames, firstMs + k * stepMs)
        }
    }

    private fun buildWindow(frames: List<Pair<Long, FloatArray>>, windowStartMs: Double): FloatArray {
        val targetLength = OnnxPredictor.SEQUENCE_LENGTH
        val featureSize = OnnxPredictor.FEATURE_SIZE
        val out = FloatArray(targetLength * featureSize) // zero-initialized -- doubles as PAD_VALUE
        val firstMs = frames.first().first
        val lastMs = frames.last().first

        for (i in 0 until targetLength) {
            val targetMs = windowStartMs + i * TARGET_FRAME_INTERVAL_MS
            if (targetMs < firstMs || targetMs > lastMs) continue // di luar rentang capture -> tetap nol (pad)

            var nearest = frames[0]
            var bestDelta = abs(frames[0].first - targetMs)
            for (f in frames) {
                val delta = abs(f.first - targetMs)
                if (delta < bestDelta) {
                    bestDelta = delta
                    nearest = f
                }
            }
            System.arraycopy(nearest.second, 0, out, i * featureSize, featureSize)
        }
        return out
    }
}
