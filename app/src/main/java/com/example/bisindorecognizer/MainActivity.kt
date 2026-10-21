package com.example.bisindorecognizer

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.bisindorecognizer.databinding.ActivityMainBinding
import com.example.bisindorecognizer.history.StatisticsActivity

/** Launcher menu: open the camera, open the statistics, or quit. */
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
            startActivity(Intent(this, StatisticsActivity::class.java))
        }

        // Also removes the task from the recent apps list
        binding.btnQuitApp.setOnClickListener {
            finishAndRemoveTask()
        }
    }
}
