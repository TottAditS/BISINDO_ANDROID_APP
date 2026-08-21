package com.example.bisindorecognizer

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.bisindorecognizer.camera.AppCameraController
import com.example.bisindorecognizer.databinding.ActivityCameraBinding
import com.example.bisindorecognizer.history.HistoryStore
import com.example.bisindorecognizer.history.RankedPrediction
import com.example.bisindorecognizer.inference.MvitPredictor
import com.example.bisindorecognizer.inference.OnnxPredictor
import com.example.bisindorecognizer.keypoints.ClipPreprocessor
import com.example.bisindorecognizer.keypoints.HandLandmarkExtractor
import com.example.bisindorecognizer.keypoints.KeypointSequenceBuilder
import com.example.bisindorecognizer.metrics.PerformanceMetrics
import com.example.bisindorecognizer.overlay.HandOverlayData
import com.example.bisindorecognizer.overlay.OverlayRenderer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * Layar kamera + prediksi live. Hidup hanya selama activity ini hidup -- ditutup lewat tombol
 * "Tutup Kamera" atau tombol Back sistem, keduanya berujung ke finish() -> onDestroy(), yang
 * melepas kamera + native resource MediaPipe/ONNX Runtime secara eksplisit (lihat releaseResources()).
 *
 * Sejak fitur perbandingan MViTv2 (lihat rencana.md di repo riset ML, PROGRESS.md untuk detail):
 * tiap sesi "Rekam" menjalankan DUA model atas rekaman yang sama -- CNNsa-BiLSTM dan MViTv2 --
 * lalu menampilkan gloss+confidence+latency dari keduanya berdampingan untuk perbandingan
 * trade-off akurasi-vs-efisiensi di laporan skripsi.
 *
 * Rekam pakai DURASI TETAP (RECORDING_DURATION_MS), bukan lagi "berhenti begitu buffer 30 frame
 * penuh" -- desain lama itu bikin sesi rekam SELALU gagal ("waktu habis") begitu ada kerja ekstra
 * per-frame yang memperlambat throughput MediaPipe (persis yang terjadi waktu fitur MViTv2
 * pertama kali ditambah -- lihat PROGRESS.md). Sekarang: tangkap SEMUA frame yang berhasil
 * terdeteksi selama durasi tetap, baru di akhir disesuaikan ke jumlah frame yang model butuhkan --
 * `KeypointSequenceBuilder` (CENTER-CROP 30 frame tengah + zero-pad, PERSIS replikasi training,
 * lihat PROGRESS.md soal kenapa bukan resample) untuk CNNsa-BiLSTM, `ClipPreprocessor` (linspace
 * sampling, replikasi training MViTv2) untuk 16 frame MViTv2.
 */
class CameraActivity : AppCompatActivity() {

    private lateinit var binding: ActivityCameraBinding
    private lateinit var cameraController: AppCameraController
    private lateinit var handLandmarkExtractor: HandLandmarkExtractor
    private lateinit var onnxPredictor: OnnxPredictor
    private lateinit var historyStore: HistoryStore
    private val mainHandler = Handler(Looper.getMainLooper())

    // MViTv2 jauh lebih berat (143MB, load pertama kali + copy dari assets ke filesDir bisa
    // beberapa detik, inferensi ratusan ms) -- dimuat SEKALI di background saat activity ini
    // dibuka, TIDAK di main thread. null berarti "belum siap" ATAU "gagal dimuat"; kedua kasus
    // itu ditangani sama (skip MViTv2 utuh untuk sesi itu, CNNsa-BiLSTM tetap jalan seperti biasa)
    // supaya kegagalan MViTv2 tidak melumpuhkan fitur inti.
    @Volatile
    private var mvitPredictor: MvitPredictor? = null

    // Dipakai untuk SEMUA kerja berat setelah sesi rekam selesai (predict CNNsa-BiLSTM, simpan
    // histori, preprocess+predict MViTv2) supaya main thread tidak pernah nge-block, DAN untuk
    // load awal MViTv2. Single-thread -> otomatis serial, tidak perlu lock tambahan di dalamnya.
    private val backgroundExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    // isRecording dibaca/ditulis dari 2 thread (klik tombol & countdown tick di main thread,
    // hasil keypoint di thread internal MediaPipe) makanya AtomicBoolean.
    private val isRecording = AtomicBoolean(false)
    private var recordingStartElapsedMs = 0L

    // SEMUA (elapsed sejak recordingStartElapsedMs, vector keypoint) yang DITERIMA MediaPipe
    // selama sesi rekam, TAPI cuma yang lolos throttle ~15fps (lihat nextCaptureFrameElapsedMs)
    // yang masuk ke sini -- lihat catatan di bawah kenapa throttle ini perlu. Timestamp ikut
    // disimpan (bukan cuma vector) karena KeypointSequenceBuilder sekarang memilih jendela 30
    // frame berbasis RENTANG WAKTU, bukan index array -- lihat dokumentasi lengkap di
    // KeypointSequenceBuilder.kt kenapa index array saja tidak cukup di device yang throughput-nya
    // di bawah 15fps. Disentuh dari thread MediaPipe (nulis) dan main thread (baca+clear di
    // awal/akhir sesi), makanya synchronized.
    private val capturedFrames = mutableListOf<Pair<Long, FloatArray>>()

    // Throttle rate SAMA dengan target_fps di PreProcessing/Scripts/frame_extractor.py (repo riset
    // ML) -- training mengasumsikan video diekstrak pada ~15fps efektif SEBELUM di-center-crop 30
    // frame. Device live berbeda-beda kecepatan throughput MediaPipe-nya (device kuat bisa kirim
    // hasil deteksi jauh lebih sering dari 15/detik, device lemah bisa lebih jarang) -- kalau SEMUA
    // hasil deteksi diterima mentah-mentah, maka 30 frame yang di-crop dari tengah merepresentasikan
    // rentang WAKTU GERAKAN ASLI yang beda-beda tergantung device (device kuat: 30 frame cuma
    // beberapa ratus ms gerakan; device lemah: 30 frame bisa 2+ detik gerakan) -- BUKAN lagi ~2
    // detik yang diasumsikan training (30 frame / 15fps). Throttle ke ~15fps di titik capture ini
    // menyamakan pacing gerakan yang di-crop nanti supaya konsisten LINTAS DEVICE, mendekati asumsi
    // training, terlepas dari seberapa cepat/lambat device memprosesnya.
    private var nextCaptureFrameElapsedMs = 0L

    // Total hasil deteksi MediaPipe yang BENERAN diterima selama sesi rekam, SEBELUM throttle di
    // atas -- murni untuk diagnostik/logging (dibandingkan capturedFrames.size yang SUDAH
    // di-throttle), supaya throughput asli tiap device kelihatan di layar Evaluasi. AtomicInteger
    // karena cuma perlu increment atomik, tidak perlu lock bareng capturedFrames.
    private val rawDetectionCount = AtomicInteger(0)

    // Untuk histori: dari seluruh durasi rekam, cuma simpan 1 komposit tiap HISTORY_FRAME_INTERVAL_MS
    // (10 total) biar hemat storage -- lihat HistoryStore untuk retensi 5 sesi terakhir. Dasarnya
    // WAKTU (elapsed sejak recordingStartElapsedMs), bukan hitungan frame, karena total frame yang
    // bakal masuk baru diketahui di akhir (durasi-driven, bukan frame-count-driven).
    private var nextHistoryFrameElapsedMs = 0L
    private val pendingHistoryFrames = mutableListOf<Bitmap>()

    // Untuk MViTv2: SEMUA frame mentah + rotasi (belum dioriented/diresize -- itu kerja berat,
    // SENGAJA ditunda ke finishRecordingAndPredict() di background thread setelah rekam selesai,
    // BUKAN dilakukan di sini per-frame. Versi awal fitur ini melakukan rotate+resize per-frame
    // LANGSUNG di thread MediaPipe -- itu akar masalah "waktu habis" yang dilaporkan user, karena
    // memperlambat throughput MediaPipe sampai 30 frame tidak sempat kekumpul. Lihat PROGRESS.md.
    private val mvitRawFrames = mutableListOf<Pair<Bitmap, Int>>()

    companion object {
        private const val TAG = "CameraActivity"
        private const val RECORDING_DURATION_MS = 5000L

        // Target 15fps -- SAMA dengan target_fps di frame_extractor.py (repo riset ML). Lihat
        // catatan di atas field nextCaptureFrameElapsedMs untuk alasan lengkapnya.
        private const val TARGET_CAPTURE_FPS = 15L
        private const val TARGET_CAPTURE_FRAME_INTERVAL_MS = 1000L / TARGET_CAPTURE_FPS

        // Sama seperti min_frames di PreProcessing/Config/dataset_config.py (repo riset ML) --
        // video training dengan < 10 frame terdeteksi di-skip sepenuhnya dari dataset, jadi model
        // tidak pernah belajar dari sequence yang sebagian besarnya nol/padding. Sesi rekam live
        // dengan frame lebih sedikit dari ini diperlakukan sama: jangan dipaksa predict.
        private const val MIN_KEYPOINT_FRAMES = 10

        private const val HISTORY_FRAMES_PER_SESSION = 10
        private const val HISTORY_FRAME_INTERVAL_MS = RECORDING_DURATION_MS / HISTORY_FRAMES_PER_SESSION
        private const val CNNSA_BILSTM_SIZE_LABEL = "14.65 MB"
        private const val MVIT_V2_SIZE_LABEL = "143 MB"
    }

    private var secondsRemaining = 0
    private val countdownRunnable = object : Runnable {
        override fun run() {
            secondsRemaining--
            if (secondsRemaining <= 0) {
                finishRecordingAndPredict()
            } else {
                binding.predictionText.text = getString(R.string.recording_countdown, secondsRemaining)
                mainHandler.postDelayed(this, 1000L)
            }
        }
    }

    private val requestCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startPipeline() else {
            binding.predictionText.text = getString(R.string.camera_permission_rationale)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        onnxPredictor = OnnxPredictor(this)
        historyStore = HistoryStore(this)
        handLandmarkExtractor = HandLandmarkExtractor(this, ::onDetectionResult)
        cameraController = AppCameraController(
            context = this,
            lifecycleOwner = this,
            previewView = binding.previewView,
            onFrame = { bitmap, rotationDegrees, timestampMs ->
                handLandmarkExtractor.submit(bitmap, rotationDegrees, timestampMs)
            }
        )

        binding.btnCloseCamera.setOnClickListener { finish() }
        binding.btnRecord.setOnClickListener { startRecording() }

        loadMvitPredictor()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startPipeline()
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startPipeline() {
        cameraController.start()
    }

    // Dijalankan sekali di background executor saat activity dibuka. btnRecord dikunci sampai ini
    // selesai (sukses ATAU gagal) supaya tidak ada sesi rekam yang mulai sebelum jelas apakah
    // MViTv2 akan ikut diproses atau di-skip untuk sesi itu.
    private fun loadMvitPredictor() {
        binding.btnRecord.isEnabled = false
        binding.predictionText.text = getString(R.string.mvit_loading)
        backgroundExecutor.execute {
            try {
                val predictor = MvitPredictor(this)
                mvitPredictor = predictor
                Log.i(TAG, "MViTv2 dimuat dalam ${predictor.loadTimeMs}ms")
                mainHandler.post {
                    binding.btnRecord.isEnabled = true
                    binding.predictionText.text = getString(R.string.prediction_placeholder)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Gagal memuat MViTv2 -- fitur perbandingan di-skip, CNNsa-BiLSTM tetap jalan", e)
                mainHandler.post {
                    binding.btnRecord.isEnabled = true
                    binding.predictionText.text = getString(R.string.mvit_load_failed)
                }
            }
        }
    }

    // Dipanggil dari klik tombol "Rekam" (main thread). Rekam durasi TETAP (bukan lagi berhenti
    // begitu 30 frame terkumpul) -- lihat doc class di atas kenapa desain lama gagal terus.
    private fun startRecording() {
        if (!isRecording.compareAndSet(false, true)) return
        synchronized(capturedFrames) {
            capturedFrames.clear()
            nextCaptureFrameElapsedMs = 0L
        }
        rawDetectionCount.set(0)
        synchronized(pendingHistoryFrames) {
            pendingHistoryFrames.clear()
            nextHistoryFrameElapsedMs = 0L
        }
        synchronized(mvitRawFrames) { mvitRawFrames.clear() }
        recordingStartElapsedMs = SystemClock.elapsedRealtime()

        // Overlay live TETAP tampil selama rekam (lihat onDetectionResult() -- sempat dimatikan
        // demi hemat CPU/throughput, tapi diaktifkan lagi 2026-08-21 atas permintaan: pengguna
        // butuh feedback visual real-time supaya bisa meragakan gestur lebih presisi mengikuti
        // titik-titik yang benar-benar terdeteksi, bukan menebak buta. Sliding-window (opsi 1,
        // lihat KeypointSequenceBuilder.alignToSlidingWindows) membuat pipeline lebih toleran
        // terhadap throughput yang sedikit turun akibat ini, jadi trade-off-nya dianggap sepadan.
        // Kalau nanti device tertentu jadi choppy lagi, opsi paling gampang: matikan lagi baris
        // update() di onDetectionResult() untuk device itu, tanpa perlu ubah bagian lain.
        binding.handOverlay.clear()

        binding.btnRecord.isEnabled = false
        binding.btnRecord.text = getString(R.string.btn_recording)

        secondsRemaining = (RECORDING_DURATION_MS / 1000L).toInt()
        binding.predictionText.text = getString(R.string.recording_countdown, secondsRemaining)
        mainHandler.postDelayed(countdownRunnable, 1000L)
    }

    // Callback hasil LIVE_STREAM MediaPipe jalan di thread internal MediaPipe (bukan main thread
    // atau thread executor CameraX). HARUS tetap murah -- lihat catatan mvitRawFrames di atas
    // soal kenapa kerja berat (rotate/resize bitmap) tidak boleh terjadi di sini.
    private fun onDetectionResult(bitmap: Bitmap, vector: FloatArray, overlay: HandOverlayData) {
        // Overlay live sekarang diupdate SELALU, termasuk selama merekam (sempat dimatikan waktu
        // rekam demi hemat CPU -- lihat catatan di startRecording() kenapa dinyalakan lagi
        // 2026-08-21) -- supaya pengguna bisa lihat titik/rangka tangannya sendiri secara
        // real-time saat meragakan gestur, bukan cuma guide box statis.
        mainHandler.post { binding.handOverlay.update(overlay) }
        if (!isRecording.get()) return
        rawDetectionCount.incrementAndGet()

        val elapsed = SystemClock.elapsedRealtime() - recordingStartElapsedMs

        // Throttle ke ~15fps (TARGET_CAPTURE_FRAME_INTERVAL_MS) supaya pacing gerakan yang nanti
        // di-center-crop KeypointSequenceBuilder konsisten lintas device -- lihat catatan panjang
        // di deklarasi nextCaptureFrameElapsedMs di atas. Timestamp (elapsed) ikut disimpan
        // bersama vector -- dipakai KeypointSequenceBuilder untuk memilih jendela berbasis waktu.
        synchronized(capturedFrames) {
            if (elapsed >= nextCaptureFrameElapsedMs) {
                nextCaptureFrameElapsedMs += TARGET_CAPTURE_FRAME_INTERVAL_MS
                capturedFrames.add(elapsed to vector)
            }
        }

        val shouldSaveHistoryFrame = synchronized(pendingHistoryFrames) {
            if (elapsed >= nextHistoryFrameElapsedMs && pendingHistoryFrames.size < HISTORY_FRAMES_PER_SESSION) {
                nextHistoryFrameElapsedMs += HISTORY_FRAME_INTERVAL_MS
                true
            } else {
                false
            }
        }
        if (shouldSaveHistoryFrame) {
            val composedFrame = OverlayRenderer.renderComposite(bitmap, overlay)
            synchronized(pendingHistoryFrames) { pendingHistoryFrames.add(composedFrame) }
        }

        // Cuma simpan referensi -- MURAH (tidak ada bitmap baru dialokasikan di sini).
        synchronized(mvitRawFrames) { mvitRawFrames.add(bitmap to overlay.rotationDegrees) }
    }

    // Dipicu countdown mencapai 0 (main thread). Snapshot semua buffer lalu lempar SELURUH kerja
    // berat (predict CNNsa-BiLSTM, simpan histori, preprocess+predict MViTv2) ke background
    // thread -- termasuk CNNsa-BiLSTM yang sebenarnya cepat, supaya main thread tidak pernah
    // menunggu apa pun. btnRecord baru aktif lagi di ujung SEMUA jalur (sukses/gagal/kosong),
    // supaya tidak ada sesi rekam baru mulai selagi sesi ini masih diproses.
    private fun finishRecordingAndPredict() {
        if (!isRecording.compareAndSet(true, false)) return

        val keypointFrames = synchronized(capturedFrames) { capturedFrames.toList().also { capturedFrames.clear() } }
        val historyFrames = synchronized(pendingHistoryFrames) { pendingHistoryFrames.toList().also { pendingHistoryFrames.clear() } }
        val mvitFrames = synchronized(mvitRawFrames) { mvitRawFrames.toList().also { mvitRawFrames.clear() } }
        val rawFrameCount = rawDetectionCount.get()

        binding.btnRecord.text = getString(R.string.btn_processing)
        binding.predictionText.text = getString(R.string.recording_processing)

        backgroundExecutor.execute {
            if (keypointFrames.size < MIN_KEYPOINT_FRAMES) {
                mainHandler.post {
                    binding.btnRecord.isEnabled = true
                    binding.btnRecord.text = getString(R.string.btn_record)
                    binding.predictionText.text = getString(R.string.recording_no_hand)
                }
                return@execute
            }

            // Opsi 1 "sliding window" (lihat PROGRESS.md 2026-08-21 dan doc KeypointSequenceBuilder):
            // coba beberapa jendela 30-frame yang tersebar di seluruh rekaman, bukan cuma satu
            // tebakan di tengah -- jendela dengan confidence top-1 tertinggi yang dipakai sebagai
            // jawaban akhir. CNNsa-BiLSTM cukup cepat (~7ms/panggilan di desktop) sehingga
            // menjalankannya DEFAULT_WINDOW_COUNT (10) kali tetap jauh di bawah ambang real-time.
            val cnnWindows = KeypointSequenceBuilder.alignToSlidingWindows(keypointFrames)
            val cnnCpuBefore = PerformanceMetrics.snapshotCpu()
            val cnnStart = System.nanoTime()
            val cnnWindowTopK = cnnWindows.map { window -> onnxPredictor.predict(window) }
            val cnnLatencyMs = (System.nanoTime() - cnnStart) / 1_000_000
            val cnnCpuPercent = PerformanceMetrics.cpuUsagePercent(cnnCpuBefore, PerformanceMetrics.snapshotCpu())
            val cnnRamKb = PerformanceMetrics.currentRamKb()

            val bestWindowIndex = cnnWindowTopK.indices.maxBy { cnnWindowTopK[it].first().second }
            val cnnSequence = cnnWindows[bestWindowIndex]
            val cnnTopK = cnnWindowTopK[bestWindowIndex]
            val (topLabelIndex, topConfidence) = cnnTopK.first()
            val cnnResultLine = getString(
                R.string.comparison_row, "CNNsa-BiLSTM", LabelMap.gloss(topLabelIndex), topConfidence * 100,
                cnnLatencyMs, CNNSA_BILSTM_SIZE_LABEL
            )
            val cnnPredictions = cnnTopK.map { (idx, conf) -> RankedPrediction(LabelMap.gloss(idx), conf) }
            // Top-1 SETIAP jendela (bukan cuma yang menang) -- disimpan buat tabel Evaluasi supaya
            // bisa dibandingkan satu-satu, sesuai permintaan: sesi X | cnn1..cnn10 confidence | ...
            val cnnWindowResults = cnnWindowTopK.map { topK ->
                val (idx, conf) = topK.first()
                RankedPrediction(LabelMap.gloss(idx), conf)
            }

            mainHandler.post {
                binding.predictionText.text = "$cnnResultLine\n${getString(R.string.mvit_processing)}"
            }

            val (mvitResultLine, mvitPredictions, mvitLatencyMs, mvitCpuPercent, mvitRamKb) = runMvitComparison(mvitFrames)

            // Histori disimpan SETELAH MViTv2 selesai (bukan langsung sesudah CNNsa-BiLSTM) supaya
            // satu entri histori memuat breakdown top-3 KEDUA model sekaligus -- boleh nunggu lebih
            // lama karena ini semua sudah di background thread, tidak ada yang nge-block UI.
            if (historyFrames.isNotEmpty()) {
                historyStore.saveSession(
                    cnnPredictions, mvitPredictions, cnnLatencyMs, mvitLatencyMs, historyFrames,
                    rawFrameCount, keypointFrames.size,
                    cnnCpuPercent, cnnRamKb, mvitCpuPercent, mvitRamKb,
                    cnnSequence, cnnWindowResults
                )
            }

            mainHandler.post {
                binding.btnRecord.isEnabled = true
                binding.btnRecord.text = getString(R.string.btn_record)
                binding.predictionText.text = "$cnnResultLine\n$mvitResultLine"
            }
        }
    }

    // 5 field -- Triple bawaan Kotlin cuma mendukung destructuring 3 komponen, jadi data class
    // sendiri supaya cpuPercent/ramKb ikut destructuring di finishRecordingAndPredict().
    private data class MvitComparisonResult(
        val resultLine: String,
        val predictions: List<RankedPrediction>,
        val latencyMs: Long?,
        val cpuPercent: Float?,
        val ramKb: Int?
    )

    // Dipanggil dari dalam backgroundExecutor (bukan main thread) -- orientForDisplay+resize yang
    // sebelumnya jalan per-frame di thread MediaPipe sekarang dilakukan di sini, sekali per sesi,
    // atas snapshot [frames]. Error/OOM MViTv2 sengaja ditangkap & dilaporkan sebagai teks (bukan
    // di-crash-kan) -- lihat rencana.md poin risiko #3: kegagalan total ITU SENDIRI temuan yang
    // sah untuk laporan skripsi. @return baris teks buat UI, daftar top-3, latency (ms), CPU%, dan
    // RAM (KB) buat disimpan ke histori/Evaluasi -- semua null/kosong kalau MViTv2 tidak
    // tersedia/gagal/tidak ada frame.
    private fun runMvitComparison(frames: List<Pair<Bitmap, Int>>): MvitComparisonResult {
        return try {
            val predictor = mvitPredictor
            when {
                predictor == null -> MvitComparisonResult(getString(R.string.mvit_unavailable), emptyList(), null, null, null)
                frames.isEmpty() -> MvitComparisonResult(getString(R.string.mvit_no_frames), emptyList(), null, null, null)
                else -> {
                    val orientedFrames = frames.map { (bitmap, rotationDegrees) ->
                        val oriented = OverlayRenderer.orientForDisplay(bitmap, rotationDegrees)
                        if (oriented.width == MvitPredictor.FRAME_SIZE && oriented.height == MvitPredictor.FRAME_SIZE) {
                            oriented
                        } else {
                            Bitmap.createScaledBitmap(oriented, MvitPredictor.FRAME_SIZE, MvitPredictor.FRAME_SIZE, true)
                        }
                    }
                    val clip = ClipPreprocessor.preprocess(orientedFrames)
                    val cpuBefore = PerformanceMetrics.snapshotCpu()
                    val start = System.nanoTime()
                    val mvitTopK = predictor.predict(clip)
                    val latencyMs = (System.nanoTime() - start) / 1_000_000
                    val cpuPercent = PerformanceMetrics.cpuUsagePercent(cpuBefore, PerformanceMetrics.snapshotCpu())
                    val ramKb = PerformanceMetrics.currentRamKb()
                    val (topLabelIndex, topConfidence) = mvitTopK.first()
                    val resultLine = getString(
                        R.string.comparison_row, "MViTv2", LabelMap.gloss(topLabelIndex),
                        topConfidence * 100, latencyMs, MVIT_V2_SIZE_LABEL
                    )
                    val predictions = mvitTopK.map { (idx, conf) -> RankedPrediction(LabelMap.gloss(idx), conf) }
                    MvitComparisonResult(resultLine, predictions, latencyMs, cpuPercent, ramKb)
                }
            }
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "MViTv2 OOM saat inferensi", e)
            MvitComparisonResult(getString(R.string.mvit_oom), emptyList(), null, null, null)
        } catch (e: Exception) {
            Log.e(TAG, "MViTv2 gagal inferensi", e)
            MvitComparisonResult(getString(R.string.mvit_error), emptyList(), null, null, null)
        }
    }

    // onDestroy() selalu dipanggil sistem saat activity ini selesai -- baik lewat tombol "Tutup
    // Kamera" (finish() eksplisit), tombol Back, maupun activity di-recreate/dibunuh sistem.
    // ProcessCameraProvider yang di-bind ke lifecycle activity ini (bindToLifecycle di
    // AppCameraController) sudah otomatis melepas kamera saat lifecycle STOPPED/DESTROYED --
    // pemanggilan eksplisit di bawah ini memastikan native resource MediaPipe HandLandmarker dan
    // sesi ONNX Runtime juga dilepas (tidak otomatis diikat ke lifecycle activity), supaya tidak
    // ada apa pun dari layar ini yang tetap jalan di background setelah activity ditutup.
    //
    // CATATAN RISIKO yang belum dimitigasi (konsisten dengan level rigor project ini di tempat
    // lain): kalau user pencet "Tutup Kamera" PAS backgroundExecutor lagi jalan (mid-predict), ada
    // race kecil antara task itu vs onnxPredictor.close()/mvitPredictor.close() di bawah ini.
    // backgroundExecutor.shutdown() cuma mencegah TASK BARU diterima, tidak menghentikan task yang
    // sedang jalan (native ONNX inference tidak reliable diinterupsi paksa).
    override fun onDestroy() {
        super.onDestroy()
        mainHandler.removeCallbacks(countdownRunnable)
        cameraController.shutdown()
        handLandmarkExtractor.close()
        onnxPredictor.close()
        mvitPredictor?.close()
        backgroundExecutor.shutdown()
    }
}
