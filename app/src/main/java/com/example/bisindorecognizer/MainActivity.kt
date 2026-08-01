package com.example.bisindorecognizer

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.bisindorecognizer.databinding.ActivityMainBinding
import com.example.bisindorecognizer.history.HistoryActivity

/** Menu screen: launcher activity, tidak menyentuh kamera sama sekali. */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnOpenCamera.setOnClickListener {
            startActivity(Intent(this, CameraActivity::class.java))
        }

        binding.btnHistory.setOnClickListener {
            startActivity(Intent(this, HistoryActivity::class.java))
        }

        // finishAndRemoveTask() menghapus task dari recent apps (bukan cuma finish() biasa yang
        // masih nyisa entry recent-apps kosong) -- ini yang paling dekat dengan "Quit" beneran di
        // Android. CameraActivity (kalau sempat dibuka) sudah pasti selesai & resource-nya
        // dilepas duluan sebelum MenuActivity ini bisa terlihat lagi, lihat CameraActivity.
        binding.btnQuitApp.setOnClickListener {
            finishAndRemoveTask()
        }
    }
}
