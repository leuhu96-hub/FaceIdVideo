plugins {
    id("com.android.application")
}

android {
    namespace = "com.example.faceid"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.faceid"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    // TFLite model must stay uncompressed so it can be memory-mapped.
    androidResources {
        noCompress += "tflite"
    }
}

dependencies {
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity:1.9.3")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.media3:media3-exoplayer:1.5.1")
    implementation("androidx.media3:media3-ui:1.5.1")
    // Bundled ML Kit face detector (model shipped inside the APK, works offline).
    implementation("com.google.mlkit:face-detection:16.1.7")
    implementation("org.tensorflow:tensorflow-lite:2.16.1")

    testImplementation("junit:junit:4.13.2")
}
