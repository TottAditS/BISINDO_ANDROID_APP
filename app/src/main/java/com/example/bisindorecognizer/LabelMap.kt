package com.example.bisindorecognizer

/** Class index to BISINDO gloss, in the order of the WL-BISINDO labels used in training. */
object LabelMap {
    private val glosses = arrayOf(
        "Air", "Belajar", "Cari", "Hari", "Ingat", "Lagi", "Maaf", "Makan", "Motor", "Saya",
        "Terima kasih", "Tuli", "Apa", "Siapa", "Kapan", "Di mana", "Mengapa", "Bagaimana",
        "Merah", "Kuning", "Hijau", "Hitam", "Dengar", "Berangkat", "Datang", "Teman", "Keluarga",
        "Rumah", "Pagi", "Siang", "Sore", "Malam"
    )

    fun gloss(index: Int): String = glosses.getOrElse(index) { "label_$index" }

    fun all(): List<String> = glosses.toList()
}
