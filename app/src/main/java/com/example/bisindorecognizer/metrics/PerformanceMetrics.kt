package com.example.bisindorecognizer.metrics

import android.os.Debug
import android.os.Process
import android.os.SystemClock

/**
 * Pengukuran RAM (PSS) + CPU usage di sekitar satu operasi (predict() satu model), dijanjikan
 * eksplisit di Ruang Lingkup Bab 1 skripsi ("penggunaan memori... serta konsumsi CPU") tapi
 * sebelumnya belum pernah diukur sama sekali -- cuma latency yang sudah tercatat.
 *
 * RAM: [Debug.getMemoryInfo] mengisi PSS (Proportional Set Size, KB) proses APLIKASI INI SAJA
 * (bukan seluruh sistem) -- metrik standar Android untuk "seberapa besar RAM yang benar-benar
 * dipakai app ini", sudah memperhitungkan native heap (ONNX Runtime, MediaPipe) yang tidak
 * kelihatan dari Kotlin/Java heap biasa. Snapshot diambil SEKALI tepat setelah predict() selesai
 * (bukan delta before/after) -- angka absolut lebih representatif buat laporan ("app memakai
 * sekian MB saat menjalankan model X") dibanding delta yang gampang bias oleh timing GC.
 *
 * CPU: [Process.getElapsedCpuTime] mengembalikan waktu CPU KUMULATIF (ms) yang sudah dipakai
 * proses ini sejak proses mulai -- diambil sebelum DAN sesudah predict(), delta-nya dibagi
 * delta waktu nyata (wall-clock) menghasilkan persentase CPU (>100% mungkin kalau kerja
 * multi-thread memakai >1 core sekaligus, itu wajar & tetap informatif). Ini best-effort
 * (tidak butuh root/profiler eksternal), bukan pengganti Android Studio Profiler yang lebih
 * presisi -- cukup untuk kebutuhan laporan skripsi, didokumentasikan sebagai keterbatasan kalau
 * perlu.
 */
object PerformanceMetrics {

    data class Snapshot(val cpuTimeMs: Long, val wallClockMs: Long)

    fun snapshotCpu(): Snapshot = Snapshot(
        cpuTimeMs = Process.getElapsedCpuTime(),
        wallClockMs = SystemClock.elapsedRealtime()
    )

    /** Persentase CPU (bisa >100% kalau >1 core terpakai bersamaan) selama rentang [before]..[after]. */
    fun cpuUsagePercent(before: Snapshot, after: Snapshot): Float {
        val wallDeltaMs = (after.wallClockMs - before.wallClockMs).toFloat()
        if (wallDeltaMs <= 0f) return 0f
        val cpuDeltaMs = (after.cpuTimeMs - before.cpuTimeMs).toFloat()
        return (cpuDeltaMs / wallDeltaMs) * 100f
    }

    /** PSS proses saat ini, dalam KB -- lihat doc object untuk kenapa PSS (bukan delta). */
    fun currentRamKb(): Int {
        val info = Debug.MemoryInfo()
        Debug.getMemoryInfo(info)
        return info.totalPss
    }
}
