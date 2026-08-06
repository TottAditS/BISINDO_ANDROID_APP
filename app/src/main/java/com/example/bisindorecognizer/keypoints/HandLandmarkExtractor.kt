package com.example.bisindorecognizer.keypoints

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log
import com.example.bisindorecognizer.overlay.HandOverlayData
import com.example.bisindorecognizer.overlay.OverlayHand
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt

/**
 * MediaPipe HandLandmarker (LIVE_STREAM). Builds the 126-float vector expected by the ONNX model:
 * left hand -> [0:63], right hand -> [63:126] (by SMOOTHED handedness label, see
 * [smoothedLabels]), each hand normalized by subtracting the wrist (landmark 0) then dividing by
 * the wrist-to-middle-MCP (landmark 9) distance. A hand that isn't detected stays all-zero.
 *
 * Reports one detection at a time via [onDetectionResult]: the bitmap that was analyzed (for
 * history-frame compositing), the 126-float model-input vector, and the raw (un-normalized)
 * 21-point landmarks per hand for HandOverlayView's bounding box/skeleton drawing. All three
 * bundled in one callback (rather than separate ones) so callers don't have to rely on call
 * order to correlate them.
 *
 * HANDEDNESS (see rencana_fix_handedness.md): MediaPipe's Left/Right `handedness()` assumes
 * selfie-mirrored input. This app uses the BACK camera (AppCameraController), which needs no
 * mirroring at any stage -- so `categoryName()` is the correct label MOST of the time. (An
 * earlier attempt kept the front camera and either mirrored the bitmap pre-MediaPipe -- broke
 * overlay geometry, mirror/rotate don't commute for 90/270deg -- or flipped the label string
 * instead; both were reverted once the camera itself switched to back, which needs neither.)
 *
 * FLICKER: MediaPipe's per-frame classification can still transiently flip for ~1s during fast
 * hand movement (observed on-device) even though it's the same physical hand -- LIVE_STREAM mode
 * gives no persistent hand ID to rely on, and a single wrong frame corrupts that frame's
 * [0:63]/[63:126] bucketing, which can corrupt the whole 30-frame CNNsa-BiLSTM sequence built from
 * it. [smoothedLabels] mitigates this with a lightweight nearest-wrist tracker (matches this
 * frame's hands to last frame's by wrist proximity) plus a majority vote over each tracked hand's
 * last [LABEL_HISTORY_SIZE] raw labels -- a single flickered frame is outvoted by the stable
 * majority, but a real, sustained handedness change (e.g. user switches which hand signs) still
 * comes through once it dominates the window.
 *
 * MISDETECTION (found 2026-08-06 reviewing Riwayat flipbook screenshots from live tests): even
 * above MediaPipe's own min_hand_detection_confidence (0.5), the landmarker occasionally locks
 * onto a non-hand region for a run of consecutive frames -- observed boxes on a shoulder/collar,
 * and BOTH detected hands labeled "Left" simultaneously (anatomically impossible, since each
 * hand's handedness is classified independently there's no built-in exclusivity). A run of
 * several bad consecutive frames is NOT something [smoothedLabels]'s 5-frame majority vote can
 * fix (it assumes flicker is the minority in its window) -- it just smooths the wrong region into
 * the tracked history. [plausibleHandIndices] filters BEFORE smoothing: drops any hand whose
 * handedness classification score is below [MIN_HANDEDNESS_CONFIDENCE] (stricter than the base
 * landmarker threshold), and de-duplicates same-label hands (keeps only the higher-confidence
 * one) -- a hand that fails this is treated as "not detected" for that frame, same zero-vector
 * convention already used for a genuinely absent hand, rather than feeding a plausibly-wrong
 * region into the model as if it were real hand data.
 */
class HandLandmarkExtractor(
    context: Context,
    private val onDetectionResult: (bitmap: Bitmap, vector: FloatArray, overlay: HandOverlayData) -> Unit
) {
    private val handLandmarker: HandLandmarker
    private val busy = AtomicBoolean(false)

    // Set right before each detectAsync() call and read back in onResult() -- safe because
    // submit()'s busy flag guarantees only one detection is in flight at a time.
    @Volatile
    private var lastRotationDegrees: Int = 0
    @Volatile
    private lateinit var lastBitmap: Bitmap

    // Handedness smoothing state -- touched ONLY from onResult() and the functions it calls
    // (buildOverlayData/buildVector126 no longer touch this directly), which all run on
    // MediaPipe's single internal result-delivery thread, sequentially -- no synchronization
    // needed despite being mutable state shared across calls.
    private val trackedHands = mutableListOf<TrackedHand>()

    private class TrackedHand(var wrist: PointF) {
        val labelHistory = ArrayDeque<String>()
    }

    init {
        val baseOptions = BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build()
        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumHands(2)
            .setMinHandDetectionConfidence(0.5f)
            .setMinHandPresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setResultListener(::onResult)
            .setErrorListener { e -> Log.e(TAG, "HandLandmarker error", e) }
            .build()
        handLandmarker = HandLandmarker.createFromOptions(context, options)
    }

    /**
     * timestampMs must be strictly increasing across calls -- use SystemClock.uptimeMillis()
     * taken at analysis time, NOT System.currentTimeMillis() (can jump due to clock sync) and NOT
     * the ImageProxy sensor timestamp (different clock base/unit, can violate monotonicity and
     * throw).
     */
    fun submit(bitmap: Bitmap, rotationDegrees: Int, timestampMs: Long) {
        // Drop frame if the previous detectAsync call hasn't finished yet -- STRATEGY_KEEP_ONLY_LATEST
        // on ImageAnalysis only controls what reaches the analyzer, not HandLandmarker's own pace.
        if (!busy.compareAndSet(false, true)) return
        lastRotationDegrees = rotationDegrees
        lastBitmap = bitmap
        handLandmarker.detectAsync(BitmapImageBuilder(bitmap).build(), timestampMs)
    }

    private fun onResult(result: HandLandmarkerResult, input: MPImage) {
        busy.set(false)
        val keptIndices = plausibleHandIndices(result)
        val labels = smoothedLabels(result, keptIndices)
        val vector = buildVector126(result, keptIndices, labels)
        val overlay = buildOverlayData(result, keptIndices, labels, input.width, input.height, lastRotationDegrees)
        onDetectionResult(lastBitmap, vector, overlay)
    }

    // Lihat "MISDETECTION" di doc class. Dipanggil SEBELUM smoothedLabels() -- hand yang tidak
    // lolos di sini tidak pernah masuk tracking/history sama sekali, diperlakukan persis seperti
    // "tidak terdeteksi" (vector nol, tidak digambar overlay) untuk frame ini.
    private fun plausibleHandIndices(result: HandLandmarkerResult): List<Int> {
        val confidences = result.handedness().map { it[0].score() }
        val labels = result.handedness().map { it[0].categoryName() }
        val candidates = confidences.indices.filter { confidences[it] >= MIN_HANDEDNESS_CONFIDENCE }

        // De-dupe: 2 kandidat dengan label sama (mustahil secara fisik) -- simpan yang
        // confidence-nya lebih tinggi saja, buang sisanya.
        return candidates
            .groupBy { labels[it] }
            .map { (_, idxs) -> idxs.maxByOrNull { confidences[it] }!! }
    }

    // Lihat doc class untuk kenapa ini ada. Dipanggil SEKALI per hasil deteksi (bukan per
    // konsumen) supaya buildOverlayData() dan buildVector126() melihat label yang SAMA persis.
    // Hanya memproses [keptIndices] (sudah lolos filter plausibilitas) -- key Map yang
    // dikembalikan adalah index ASLI di result.landmarks()/handedness(), bukan posisi di keptIndices.
    private fun smoothedLabels(result: HandLandmarkerResult, keptIndices: List<Int>): Map<Int, String> {
        val wrists = keptIndices.associateWith { PointF(result.landmarks()[it][0].x(), result.landmarks()[it][0].y()) }
        val rawLabels = keptIndices.associateWith { result.handedness()[it][0].categoryName() }

        val usedTracked = mutableSetOf<Int>()
        val matched = keptIndices.associateWith { idx ->
            var bestIdx = -1
            var bestDist = Float.MAX_VALUE
            for (j in trackedHands.indices) {
                if (j in usedTracked) continue
                val d = distance(wrists.getValue(idx), trackedHands[j].wrist)
                if (d < bestDist) {
                    bestDist = d
                    bestIdx = j
                }
            }
            if (bestIdx != -1 && bestDist < MAX_TRACKING_DISTANCE) {
                usedTracked.add(bestIdx)
                trackedHands[bestIdx]
            } else {
                null
            }
        }

        val nextTrackedHands = mutableListOf<TrackedHand>()
        val smoothed = keptIndices.associateWith { idx ->
            val tracked = matched.getValue(idx) ?: TrackedHand(wrists.getValue(idx))
            tracked.wrist = wrists.getValue(idx)
            tracked.labelHistory.addLast(rawLabels.getValue(idx))
            if (tracked.labelHistory.size > LABEL_HISTORY_SIZE) tracked.labelHistory.removeFirst()
            nextTrackedHands.add(tracked)
            tracked.labelHistory.groupingBy { it }.eachCount().maxByOrNull { it.value }!!.key
        }
        trackedHands.clear()
        trackedHands.addAll(nextTrackedHands)
        return smoothed
    }

    private fun distance(a: PointF, b: PointF): Float {
        val dx = a.x - b.x
        val dy = a.y - b.y
        return sqrt(dx * dx + dy * dy)
    }

    private fun buildOverlayData(
        result: HandLandmarkerResult,
        keptIndices: List<Int>,
        labels: Map<Int, String>,
        imageWidth: Int,
        imageHeight: Int,
        rotationDegrees: Int
    ): HandOverlayData {
        val hands = keptIndices.map { i ->
            val points = result.landmarks()[i].map { PointF(it.x(), it.y()) }
            OverlayHand(labels.getValue(i), points)
        }
        return HandOverlayData(hands, imageWidth, imageHeight, rotationDegrees)
    }

    private fun buildVector126(result: HandLandmarkerResult, keptIndices: List<Int>, labels: Map<Int, String>): FloatArray {
        val vector = FloatArray(126)
        for (i in keptIndices) {
            val offset = if (labels.getValue(i) == "Left") 0 else 63
            val lm = result.landmarks()[i]
            val wrist = lm[0]
            val midMcp = lm[9]
            val dx = midMcp.x() - wrist.x()
            val dy = midMcp.y() - wrist.y()
            val dz = midMcp.z() - wrist.z()
            val scale = sqrt(dx * dx + dy * dy + dz * dz)
            for (j in 0 until 21) {
                var x = lm[j].x() - wrist.x()
                var y = lm[j].y() - wrist.y()
                var z = lm[j].z() - wrist.z()
                if (scale >= 1e-6f) {
                    x /= scale; y /= scale; z /= scale
                }
                vector[offset + j * 3] = x
                vector[offset + j * 3 + 1] = y
                vector[offset + j * 3 + 2] = z
            }
        }
        return vector
    }

    fun close() {
        handLandmarker.close()
    }

    companion object {
        private const val TAG = "HandLandmarkExtractor"

        // Berapa banyak label mentah terakhir (per tangan yang di-track) dipakai buat majority
        // vote -- makin besar makin tahan flicker, tapi makin lambat merespons pertukaran tangan
        // yang BENERAN terjadi (mis. user ganti tangan yang bergestur). Nilai awal, belum ditest.
        private const val LABEL_HISTORY_SIZE = 5

        // Jarak (koordinat ternormalisasi [0,1]) maksimum antar wrist 2 frame berurutan supaya
        // masih dianggap tangan fisik yang sama -- gerakan tangan lebih jauh dari ini di 1 frame
        // dianggap tangan baru (history mulai dari nol lagi). Nilai awal, belum ditest di device.
        private const val MAX_TRACKING_DISTANCE = 0.25f

        // Ambang confidence handedness classification -- SENGAJA lebih ketat dari
        // min_hand_detection_confidence (0.5) yang dipakai HandLandmarkerOptions di atas. Lihat
        // "MISDETECTION" di doc class: box yang nyasar ke bahu/wajah yang diamati di Riwayat masih
        // lolos ambang 0.5 milik landmarker sendiri -- ambang tambahan ini di titik BERBEDA
        // (skor klasifikasi Left/Right per hand, bukan skor deteksi awal) buat nyaring kandidat
        // yang lolos deteksi tapi klasifikasi handedness-nya sendiri tidak yakin. Nilai awal,
        // belum ditest di device -- turunkan kalau tangan asli malah sering ke-reject (tidak
        // pernah kelihatan overlay-nya), naikkan kalau box salah masih lolos.
        private const val MIN_HANDEDNESS_CONFIDENCE = 0.8f
    }
}
