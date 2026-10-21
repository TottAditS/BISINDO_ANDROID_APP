package com.example.bisindorecognizer.metrics

import android.os.Debug
import android.os.Process
import android.os.SystemClock

/**
 * CPU and memory measurements around one model call.
 *
 * Memory: the PSS (proportional set size) of this app's process in KB, taken once right after
 * predict(). PSS includes native memory such as ONNX Runtime and MediaPipe.
 *
 * CPU: process CPU time divided by wall-clock time between two snapshots, in percent. Values above
 * 100% mean more than one core was busy. This is a lightweight estimate, not a profiler.
 */
object PerformanceMetrics {

    data class Snapshot(val cpuTimeMs: Long, val wallClockMs: Long)

    fun snapshotCpu(): Snapshot = Snapshot(
        cpuTimeMs = Process.getElapsedCpuTime(),
        wallClockMs = SystemClock.elapsedRealtime()
    )

    /** CPU usage in percent between [before] and [after]; above 100% when several cores are busy. */
    fun cpuUsagePercent(before: Snapshot, after: Snapshot): Float {
        val wallDeltaMs = (after.wallClockMs - before.wallClockMs).toFloat()
        if (wallDeltaMs <= 0f) return 0f
        val cpuDeltaMs = (after.cpuTimeMs - before.cpuTimeMs).toFloat()
        return (cpuDeltaMs / wallDeltaMs) * 100f
    }

    /** Current PSS of the process in KB. */
    fun currentRamKb(): Int {
        val info = Debug.MemoryInfo()
        Debug.getMemoryInfo(info)
        return info.totalPss
    }
}
