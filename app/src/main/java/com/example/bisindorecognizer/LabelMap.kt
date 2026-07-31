package com.example.bisindorecognizer

/**
 * Index -> gloss BISINDO, urutan sesuai training (32 kelas). Sumber: tabel mapping yang
 * disediakan user (tidak ada di file config repo riset ML -- lihat README bagian
 * "Keterbatasan yang sudah dikonfirmasi").
 */
object LabelMap {
    private val glosses = arrayOf(
        "Air", "Belajar", "Cari", "Hari", "Ingat", "Lagi", "Maaf", "Makan", "Motor", "Saya",
        "Terima kasih", "Tuli", "Apa", "Siapa", "Kapan", "Di mana", "Mengapa", "Bagaimana",
        "Merah", "Kuning", "Hijau", "Hitam", "Dengar", "Berangkat", "Datang", "Teman", "Keluarga",
        "Rumah", "Pagi", "Siang", "Sore", "Malam"
    )

    fun gloss(index: Int): String = glosses.getOrElse(index) { "label_$index" }
}
