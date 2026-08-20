package com.example.bisindorecognizer.gif

import android.graphics.Bitmap
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.OutputStream

/**
 * Encoder GIF animasi self-contained (TIDAK menambah dependency Gradle baru -- konsisten dengan
 * `history/LatencyChartView.kt` yang juga sengaja Canvas manual, bukan library chart) -- dipakai
 * untuk fitur "Unduh GIF" di HistoryPlaybackActivity: menggabungkan 10 frame overlay-baked yang
 * SUDAH tersimpan di Riwayat jadi satu file .gif yang bisa dibagikan/dilampirkan di laporan.
 *
 * Port Kotlin dari algoritma GIF encoder + NeuQuant color quantizer klasik karya Kevin Weiner /
 * Anthony Dekker (keduanya public domain, dipakai luas di banyak project open-source) -- diadaptasi
 * untuk baca piksel dari [Bitmap] Android langsung (bukan `java.awt.image.BufferedImage`, yang
 * tidak tersedia di runtime Android) via `Bitmap.getPixels()`.
 *
 * GIF cuma dukung 256 warna per palet -- NeuQuant mereduksi tiap frame RGB 24-bit ke <=256 warna
 * lewat jaringan neural sederhana (self-organizing map atas ruang warna), lalu LZW meng-kompresi
 * indeks piksel hasil kuantisasi itu. Detail algoritma TIDAK diubah dari referensi aslinya --
 * cuma sumber piksel & output stream yang disesuaikan ke Android.
 */
class GifEncoder(private val out: OutputStream) {
    private var width = 0
    private var height = 0
    private var delayMs = 200
    private var repeat = 0 // 0 = loop selamanya
    private var started = false
    private var firstFrame = true

    fun setDelay(ms: Int) {
        delayMs = ms
    }

    fun setRepeat(count: Int) {
        repeat = count
    }

    @Throws(IOException::class)
    fun start() {
        out.write("GIF89a".toByteArray(Charsets.US_ASCII))
        started = true
    }

    @Throws(IOException::class)
    fun addFrame(bitmap: Bitmap) {
        if (!started) throw IllegalStateException("call start() first")
        width = bitmap.width
        height = bitmap.height

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // NeuQuant butuh byte[] RGB berurutan (bukan ARGB int Android).
        val rgb = ByteArray(width * height * 3)
        for (i in pixels.indices) {
            val p = pixels[i]
            rgb[i * 3] = ((p shr 16) and 0xFF).toByte()
            rgb[i * 3 + 1] = ((p shr 8) and 0xFF).toByte()
            rgb[i * 3 + 2] = (p and 0xFF).toByte()
        }

        val quant = NeuQuant(rgb, rgb.size, 10) // sampleFactor 10 -- cukup cepat, kualitas wajar
        val colorTab = quant.process() // palet 256 warna, urutan RGB berurutan
        val indexedPixels = ByteArray(pixels.size)
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            indexedPixels[i] = quant.map(r, g, b).toByte()
        }

        if (firstFrame) {
            writeLogicalScreenDescriptor()
            writeGlobalColorTable(colorTab)
            if (repeat >= 0) writeNetscapeExtension()
            firstFrame = false
        }

        writeGraphicControlExtension()
        writeImageDescriptor()
        writePixels(indexedPixels)
    }

    @Throws(IOException::class)
    fun finish() {
        out.write(0x3B) // GIF trailer
    }

    private fun writeLogicalScreenDescriptor() {
        writeShort(width)
        writeShort(height)
        // Global color table flag=1, color resolution=7, sort=0, size=7 (2^(7+1)=256 warna)
        out.write(0xF7)
        out.write(0) // background color index
        out.write(0) // pixel aspect ratio
    }

    private fun writeGlobalColorTable(colorTab: ByteArray) {
        out.write(colorTab)
        val padding = 256 * 3 - colorTab.size
        if (padding > 0) out.write(ByteArray(padding))
    }

    private fun writeNetscapeExtension() {
        out.write(0x21) // extension introducer
        out.write(0xFF) // application extension label
        out.write(11) // block size
        out.write("NETSCAPE2.0".toByteArray(Charsets.US_ASCII))
        out.write(3) // sub-block size
        out.write(1) // loop sub-block id
        writeShort(repeat) // loop count, 0 = infinite
        out.write(0) // block terminator
    }

    private fun writeGraphicControlExtension() {
        out.write(0x21)
        out.write(0xF9)
        out.write(4) // block size
        out.write(0x00) // no transparency, no disposal specified
        writeShort(delayMs / 10) // delay dalam 1/100 detik
        out.write(0) // transparent color index (tidak dipakai)
        out.write(0) // block terminator
    }

    private fun writeImageDescriptor() {
        out.write(0x2C) // image separator
        writeShort(0) // left
        writeShort(0) // top
        writeShort(width)
        writeShort(height)
        out.write(0) // no local color table, not interlaced
    }

    private fun writePixels(indexedPixels: ByteArray) {
        val lzw = LzwEncoder(width, height, indexedPixels, 8)
        lzw.encode(out)
    }

    private fun writeShort(value: Int) {
        out.write(value and 0xFF)
        out.write((value shr 8) and 0xFF)
    }
}

/**
 * Kuantisasi warna NeuQuant (Anthony Dekker, public domain) -- mereduksi gambar 24-bit jadi
 * <=256 warna via self-organizing neural network di ruang warna RGB. Port langsung dari
 * implementasi referensi, cuma diringkas (tanpa dependency java.awt).
 */
private class NeuQuant(private val pixels: ByteArray, private val pixelCount: Int, sampleFactorInput: Int) {
    companion object {
        private const val NETSIZE = 256
        private const val PRIME1 = 499
        private const val PRIME2 = 491
        private const val PRIME3 = 487
        private const val PRIME4 = 503
        private const val MINPICTUREBYTES = 3 * PRIME4
        private const val MAXNETPOS = NETSIZE - 1
        private const val NETBIASSHIFT = 4
        private const val NCYCLES = 100
        private const val INTBIASSHIFT = 16
        private const val INTBIAS = 1 shl INTBIASSHIFT
        private const val INITRAD = NETSIZE shr 3
        private const val RADIUSBIASSHIFT = 6
        private const val RADIUSBIAS = 1 shl RADIUSBIASSHIFT
        private const val INITRADIUS = INITRAD * RADIUSBIAS
        private const val RADIUSDEC = 30
        private const val ALPHABIASSHIFT = 10
        private const val INITALPHA = 1 shl ALPHABIASSHIFT
        private const val RADBIASSHIFT = 8
        private const val RADBIAS = 1 shl RADBIASSHIFT
        private const val ALPHARADBSHIFT = ALPHABIASSHIFT + RADBIASSHIFT
        private const val ALPHARADBIAS = 1 shl ALPHARADBSHIFT
    }

    private val network = Array(NETSIZE) { DoubleArray(4) }
    private val netindex = IntArray(256)
    private val bias = IntArray(NETSIZE)
    private val freq = IntArray(NETSIZE)
    private val radpower = IntArray(INITRAD)
    private val sampleFactor = sampleFactorInput.coerceIn(1, 30)

    init {
        for (i in 0 until NETSIZE) {
            val v = (i shl (NETBIASSHIFT + 8)) / NETSIZE
            network[i][0] = v.toDouble()
            network[i][1] = v.toDouble()
            network[i][2] = v.toDouble()
            freq[i] = INTBIAS / NETSIZE
            bias[i] = 0
        }
    }

    fun process(): ByteArray {
        learn()
        unbiasnet()
        inxbuild()
        return colorMap()
    }

    private fun colorMap(): ByteArray {
        val map = ByteArray(NETSIZE * 3)
        val index = IntArray(NETSIZE)
        for (i in 0 until NETSIZE) index[network[i][3].toInt()] = i
        var k = 0
        for (i in 0 until NETSIZE) {
            val j = index[i]
            map[k++] = network[j][0].toInt().toByte()
            map[k++] = network[j][1].toInt().toByte()
            map[k++] = network[j][2].toInt().toByte()
        }
        return map
    }

    fun map(r: Int, g: Int, b: Int): Int {
        var bestd = Int.MAX_VALUE
        var best = -1
        var i = netindex[g]
        var j = i - 1
        while (i < NETSIZE || j >= 0) {
            if (i < NETSIZE) {
                val p = network[i]
                var dist = p[1].toInt() - g
                if (dist >= bestd) {
                    i = NETSIZE
                } else {
                    if (dist < 0) dist = -dist
                    var a = p[0].toInt() - r
                    if (a < 0) a = -a
                    dist += a
                    if (dist < bestd) {
                        a = p[2].toInt() - b
                        if (a < 0) a = -a
                        dist += a
                        if (dist < bestd) {
                            bestd = dist
                            best = p[3].toInt()
                        }
                    }
                    i++
                }
            }
            if (j >= 0) {
                val p = network[j]
                var dist = g - p[1].toInt()
                if (dist >= bestd) {
                    j = -1
                } else {
                    if (dist < 0) dist = -dist
                    var a = p[0].toInt() - r
                    if (a < 0) a = -a
                    dist += a
                    if (dist < bestd) {
                        a = p[2].toInt() - b
                        if (a < 0) a = -a
                        dist += a
                        if (dist < bestd) {
                            bestd = dist
                            best = p[3].toInt()
                        }
                    }
                    j--
                }
            }
        }
        return best
    }

    private fun unbiasnet() {
        for (i in 0 until NETSIZE) {
            network[i][0] = network[i][0].toInt().toDouble()
            network[i][1] = network[i][1].toInt().toDouble()
            network[i][2] = network[i][2].toInt().toDouble()
            network[i][3] = i.toDouble()
        }
    }

    private fun inxbuild() {
        var previouscol = 0
        var startpos = 0
        for (i in 0 until NETSIZE) {
            val p = network[i]
            var smallpos = i
            var smallval = p[1].toInt()
            for (j in i + 1 until NETSIZE) {
                val q = network[j]
                if (q[1].toInt() < smallval) {
                    smallpos = j
                    smallval = q[1].toInt()
                }
            }
            val q = network[smallpos]
            if (i != smallpos) {
                var tmp = q[0]; q[0] = p[0]; p[0] = tmp
                tmp = q[1]; q[1] = p[1]; p[1] = tmp
                tmp = q[2]; q[2] = p[2]; p[2] = tmp
                tmp = q[3]; q[3] = p[3]; p[3] = tmp
            }
            if (smallval != previouscol) {
                netindex[previouscol] = (startpos + i) shr 1
                for (j in previouscol + 1 until smallval) netindex[j] = i
                previouscol = smallval
                startpos = i
            }
        }
        netindex[previouscol] = (startpos + MAXNETPOS) shr 1
        for (j in previouscol + 1 until 256) netindex[j] = MAXNETPOS
    }

    private fun altersingle(alpha: Int, i: Int, b: Int, g: Int, r: Int) {
        val n = network[i]
        n[0] -= (alpha * (n[0] - r)) / INITALPHA
        n[1] -= (alpha * (n[1] - g)) / INITALPHA
        n[2] -= (alpha * (n[2] - b)) / INITALPHA
    }

    /** [rad] neuron neighbors of [i] pulled toward (r,g,b), weight tapering off via [radpower] (built in learn()). */
    private fun alterneigh(rad: Int, i: Int, b: Int, g: Int, r: Int) {
        var lo = i - rad
        if (lo < -1) lo = -1
        var hi = i + rad
        if (hi > NETSIZE) hi = NETSIZE
        var j = i + 1
        var k = i - 1
        var m = 1
        while (j < hi || k > lo) {
            val a = radpower[m]
            if (j < hi) {
                val p = network[j]
                p[0] -= (a * (p[0] - r)) / ALPHARADBIAS
                p[1] -= (a * (p[1] - g)) / ALPHARADBIAS
                p[2] -= (a * (p[2] - b)) / ALPHARADBIAS
                j++
            }
            if (k > lo) {
                val p = network[k]
                p[0] -= (a * (p[0] - r)) / ALPHARADBIAS
                p[1] -= (a * (p[1] - g)) / ALPHARADBIAS
                p[2] -= (a * (p[2] - b)) / ALPHARADBIAS
                k--
            }
            m++
        }
    }

    private fun buildRadpower(rad: Int, alpha: Int) {
        for (m in 0 until rad) {
            radpower[m] = (alpha * ((rad * rad - m * m) * RADBIAS)) / (rad * rad)
        }
    }

    private fun contest(b: Int, g: Int, r: Int): Int {
        var bestd = Int.MAX_VALUE
        var bestbiasd = bestd
        var bestpos = -1
        var bestbiaspos = bestpos
        for (i in 0 until NETSIZE) {
            val n = network[i]
            var dist = n[0].toInt() - r
            if (dist < 0) dist = -dist
            var a = n[1].toInt() - g
            if (a < 0) a = -a
            dist += a
            a = n[2].toInt() - b
            if (a < 0) a = -a
            dist += a
            if (dist < bestd) {
                bestd = dist
                bestpos = i
            }
            val biasdist = dist - ((bias[i]) shr (INTBIASSHIFT - NETBIASSHIFT))
            if (biasdist < bestbiasd) {
                bestbiasd = biasdist
                bestbiaspos = i
            }
            freq[i] -= freq[i] shr NETBIASSHIFT
            bias[i] += freq[i] shl (INTBIASSHIFT - NETBIASSHIFT)
        }
        freq[bestpos] += 1 shl NETBIASSHIFT
        bias[bestpos] -= 1 shl INTBIASSHIFT
        return bestbiaspos
    }

    private fun learn() {
        val lengthcount = pixelCount
        val effectiveSampleFactor = if (lengthcount < MINPICTUREBYTES) 1 else sampleFactor
        val alphadec = 30 + (effectiveSampleFactor - 1) / 3
        val samplepixels = lengthcount / (3 * effectiveSampleFactor)
        var delta = samplepixels / NCYCLES
        if (delta == 0) delta = 1
        var alpha = INITALPHA
        var radius = INITRADIUS
        var rad = radius shr RADIUSBIASSHIFT
        if (rad <= 1) rad = 0
        if (rad > 0) buildRadpower(rad, alpha)

        var pos = 0
        val step = if (lengthcount < MINPICTUREBYTES) 3 else if (lengthcount % PRIME1 != 0) 3 * PRIME1 else if (lengthcount % PRIME2 != 0) 3 * PRIME2 else if (lengthcount % PRIME3 != 0) 3 * PRIME3 else 3 * PRIME4

        var i = 0
        while (i < samplepixels) {
            val b = (pixels[pos + 2].toInt() and 0xFF)
            val g = (pixels[pos + 1].toInt() and 0xFF)
            val r = (pixels[pos].toInt() and 0xFF)
            val j = contest(b, g, r)
            altersingle(alpha, j, b, g, r)
            if (rad != 0) alterneigh(rad, j, b, g, r)

            pos += step
            if (pos >= lengthcount) pos -= lengthcount

            i++
            if (i % delta == 0) {
                alpha -= alpha / alphadec
                radius -= radius / RADIUSDEC
                rad = radius shr RADIUSBIASSHIFT
                if (rad <= 1) rad = 0
                if (rad > 0) buildRadpower(rad, alpha)
            }
        }
    }
}

/**
 * Kompresi LZW varian GIF (berdasarkan algoritma UNIX `compress`, diadaptasi untuk GIF oleh
 * Steve Wilhite -- public domain, sama seperti versi Kevin Weiner). Menerima indeks piksel hasil
 * kuantisasi NeuQuant, menulis blok data ter-kompresi langsung ke [OutputStream].
 */
private class LzwEncoder(
    private val imgW: Int,
    private val imgH: Int,
    private val pixels: ByteArray,
    private val colorDepth: Int
) {
    private val initCodeSize = maxOf(2, colorDepth)
    private var curPixel = 0
    private var curAccum = 0
    private var curBits = 0
    private val masks = intArrayOf(
        0x0000, 0x0001, 0x0003, 0x0007, 0x000F, 0x001F, 0x003F, 0x007F,
        0x00FF, 0x01FF, 0x03FF, 0x07FF, 0x0FFF, 0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF
    )
    private var nBits = 0
    private var maxbits = 12
    private var maxcode = 0
    private var maxmaxcode = 1 shl 12
    private var hashSize = 5003
    private var htab = IntArray(hashSize)
    private var codetab = IntArray(hashSize)
    private var hsize = hashSize
    private var freeEnt = 0
    private var clearFlg = false
    private var gInitBits = 0
    private var clearCode = 0
    private var eofCode = 0
    private var accum = ByteArray(256)
    private var accumCount = 0

    private fun maxcode(bits: Int) = (1 shl bits) - 1

    fun encode(out: OutputStream) {
        out.write(colorDepth)
        val pixelCount = imgW * imgH
        curPixel = 0
        compress(colorDepth + 1, out, pixelCount)
        out.write(0) // block terminator
    }

    private fun compress(initBits: Int, out: OutputStream, pixelCount: Int) {
        gInitBits = initBits
        clearFlg = false
        nBits = gInitBits
        maxcode = maxcode(nBits)
        clearCode = 1 shl (initBits - 1)
        eofCode = clearCode + 1
        freeEnt = clearCode + 2
        accumCount = 0
        curAccum = 0
        curBits = 0

        var ent = nextPixel(pixelCount)
        var hshift = 0
        var fcode = hsize
        while (fcode < 65536) {
            fcode *= 2
            hshift++
        }
        hshift = 8 - hshift
        val hsizeReg = hsize
        clearHash(hsizeReg)
        output(clearCode, out)

        outerLoop@ while (true) {
            val c = nextPixel(pixelCount)
            if (c == -1) break
            val fcodeVal = (c shl maxbits) + ent
            var i = (c shl hshift) xor ent
            if (htab[i] == fcodeVal) {
                ent = codetab[i]
                continue
            } else if (htab[i] >= 0) {
                var disp = hsizeReg - i
                if (i == 0) disp = 1
                do {
                    i -= disp
                    if (i < 0) i += hsizeReg
                    if (htab[i] == fcodeVal) {
                        ent = codetab[i]
                        continue@outerLoop
                    }
                } while (htab[i] >= 0)
            }
            output(ent, out)
            ent = c
            if (freeEnt < maxmaxcode) {
                codetab[i] = freeEnt++
                htab[i] = fcodeVal
            } else {
                clearHash(hsizeReg)
                freeEnt = clearCode + 2
                clearFlg = true
                output(clearCode, out)
                nBits = gInitBits
                maxcode = maxcode(nBits)
            }
        }
        output(ent, out)
        output(eofCode, out)
    }

    private fun nextPixel(pixelCount: Int): Int {
        if (curPixel >= pixelCount) return -1
        val pix = pixels[curPixel].toInt() and 0xFF
        curPixel++
        return pix
    }

    private fun clearHash(hsizeReg: Int) {
        htab = IntArray(hsizeReg) { -1 }
    }

    private fun output(code: Int, out: OutputStream) {
        curAccum = curAccum or (code shl curBits)
        curBits += nBits
        while (curBits >= 8) {
            addByte((curAccum and 0xFF).toByte(), out)
            curAccum = curAccum shr 8
            curBits -= 8
        }
        if (freeEnt > maxcode || clearFlg) {
            if (clearFlg) {
                nBits = gInitBits
                maxcode = maxcode(nBits)
                clearFlg = false
            } else {
                nBits++
                maxcode = if (nBits == maxbits) maxmaxcode else maxcode(nBits)
            }
        }
        if (code == eofCode) {
            while (curBits > 0) {
                addByte((curAccum and 0xFF).toByte(), out)
                curAccum = curAccum shr 8
                curBits -= 8
            }
            flushBytes(out)
        }
    }

    private fun addByte(b: Byte, out: OutputStream) {
        accum[accumCount++] = b
        if (accumCount >= 254) flushBytes(out)
    }

    private fun flushBytes(out: OutputStream) {
        if (accumCount > 0) {
            out.write(accumCount)
            out.write(accum, 0, accumCount)
            accumCount = 0
        }
    }
}
