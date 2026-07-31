package com.example.bisindorecognizer

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.bisindorecognizer.camera.AppCameraController
import com.example.bisindorecognizer.databinding.ActivityCameraBinding
import com.example.bisindorecognizer.inference.OnnxPredictor
import com.example.bisindorecognizer.keypoints.HandLandmarkExtractor
import com.example.bisindorecognizer.keypoints.SequenceBuffer

/**
 * Layar kamera + prediksi live. Hidup hanya selama activity ini hidup -- ditutup lewat tombol
 * "Tutup Kamera" atau tombol Back sistem, keduanya berujung ke finish() -> onDestroy(), yang
 * melepas kamera + native resource MediaPipe/ONNX Runtime secara eksplisit (lihat releaseResources()).
 */
class CameraActivity : AppCompatActivity() {

    private lateinit var binding: ActivityCameraBinding
    private lateinit var cameraController: AppCameraController
    private lateinit var handLandmarkExtractor: HandLandmarkExtractor
    private lateinit var onnxPredictor: OnnxPredictor
    private val sequenceBuffer = SequenceBuffer()
    private val mainHandler = Handler(Looper.getMainLooper())

    // Jalankan inference tiap beberapa frame saja (bukan tiap push) supaya tidak terlalu berat
    // di Poco M4 Pro (CPU-only, tanpa NPU yang bisa diandalkan).
    private var frameCounter = 0
    private val inferenceIntervalFrames = 5

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
        handLandmarkExtractor = HandLandmarkExtractor(this, ::onKeypointVector)
        cameraController = AppCameraController(
            context = this,
            lifecycleOwner = this,
            previewView = binding.previewView,
            onFrame = { bitmap, _, timestampMs -> handLandmarkExtractor.submit(bitmap, timestampMs) }
        )

        binding.btnCloseCamera.setOnClickListener { finish() }

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

    // Callback hasil LIVE_STREAM MediaPipe jalan di thread internal MediaPipe (bukan main thread
    // atau thread executor CameraX). push()+predict() cukup ringan untuk jalan langsung di situ,
    // tapi update TextView WAJIB dipindah ke main thread.
    private fun onKeypointVector(vector: FloatArray) {
        sequenceBuffer.push(vector)
        if (!sequenceBuffer.isFull()) return

        frameCounter++
        if (frameCounter % inferenceIntervalFrames != 0) return

        val (labelIndex, confidence) = onnxPredictor.predict(sequenceBuffer.toFlatFloatArray())
        val text = "${LabelMap.gloss(labelIndex)} (${"%.1f".format(confidence * 100)}%)"
        mainHandler.post { binding.predictionText.text = text }
    }

    // onDestroy() selalu dipanggil sistem saat activity ini selesai -- baik lewat tombol "Tutup
    // Kamera" (finish() eksplisit), tombol Back, maupun activity di-recreate/dibunuh sistem.
    // ProcessCameraProvider yang di-bind ke lifecycle activity ini (bindToLifecycle di
    // AppCameraController) sudah otomatis melepas kamera saat lifecycle STOPPED/DESTROYED --
    // pemanggilan eksplisit di bawah ini memastikan native resource MediaPipe HandLandmarker dan
    // sesi ONNX Runtime juga dilepas (tidak otomatis diikat ke lifecycle activity), supaya tidak
    // ada apa pun dari layar ini yang tetap jalan di background setelah activity ditutup.
    override fun onDestroy() {
        super.onDestroy()
        cameraController.shutdown()
        handLandmarkExtractor.close()
        onnxPredictor.close()
    }
}
