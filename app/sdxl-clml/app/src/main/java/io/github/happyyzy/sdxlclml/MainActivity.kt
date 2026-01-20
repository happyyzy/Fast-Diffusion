package io.github.happyyzy.sdxlclml

import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import java.io.BufferedReader
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import kotlin.math.max
import kotlin.math.min
import kotlinx.coroutines.delay
import org.json.JSONObject

private const val SDXL_WIDTH = 1024
private const val SDXL_HEIGHT = 1024
private const val SD15_WIDTH = 512
private const val SD15_HEIGHT = 512
private const val BACKEND_PORT_SDXL = 8081
private const val BACKEND_PORT_SD15 = 8082
private const val BACKEND_HEALTH_TIMEOUT_MS = 120000L
private const val BACKEND_GENERATE_TIMEOUT_MS = 900000
private const val MAX_LOG_CHARS = 8000
private const val HF_REPO = "zhiyuanasad/fast-diffusion-weights"
private const val HF_API = "https://huggingface.co/api/models/$HF_REPO"
private const val HF_RESOLVE = "https://huggingface.co/$HF_REPO/resolve/main/"
private const val SDXL_MIN_MEM_GB = 14.5

private enum class ModelKind {
    SDXL,
    SD15,
}

private data class HfFile(
    val path: String,
    val size: Long,
    val sha256: String,
)

private data class EnvStatus(
    val openClOk: Boolean,
    val totalMemGb: Double,
    val sdxlOk: Boolean,
)

private data class RuntimePaths(
    val runtimeDir: File,
    val binDir: File,
    val mnnFuseDir: File,
    val workDir: File,
    val baseDir: File,
    val externalDir: File,
    val weightsDir: File,
    val sd15WeightsDir: File,
    val mnnClipDir: File,
    val tokensDir: File,
    val clipL: File,
    val clipG: File,
    val tokenizerJson: File,
    val tokenizer2Json: File,
    val sd15TokenizerJson: File,
    val backendBin: File,
    val sd15BackendBin: File,
    val clipBin: File,
)

private fun getTotalMemGb(context: Context): Double {
    val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    val info = ActivityManager.MemoryInfo()
    am.getMemoryInfo(info)
    return info.totalMem.toDouble() / (1024.0 * 1024.0 * 1024.0)
}

private fun hasOpenClLib(): Boolean {
    return File("/vendor/lib64/libOpenCL.so").exists() ||
        File("/system/lib64/libOpenCL.so").exists()
}

private fun computeEnvStatus(context: Context, mnnFuseDir: File): EnvStatus {
    val totalMemGb = getTotalMemGb(context)
    val openClOk = hasOpenClLib() && File(mnnFuseDir, "libMNN_CL.so").exists()
    val sdxlOk = openClOk && totalMemGb >= SDXL_MIN_MEM_GB
    return EnvStatus(openClOk = openClOk, totalMemGb = totalMemGb, sdxlOk = sdxlOk)
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                MainScreen()
            }
        }
    }
}

@Composable
private fun MainScreen() {
    val context = LocalContext.current
    val mainHandler = remember { Handler(Looper.getMainLooper()) }

    var modelKind by remember { mutableStateOf(ModelKind.SDXL) }
    var promptText by remember { mutableStateOf("portrait photo, ultra detailed, 35mm") }
    var negativeText by remember { mutableStateOf("") }
    var stepsText by remember { mutableStateOf("20") }
    var cfgText by remember { mutableStateOf("7.5") }
    var seedText by remember { mutableStateOf("0") }
    var earlyKText by remember { mutableStateOf("2") }
    var decodeX0 by remember { mutableStateOf(true) }
    var useExternalClip by remember { mutableStateOf(true) }

    var statusText by remember { mutableStateOf("Idle") }
    var progressText by remember { mutableStateOf("") }
    var outputText by remember { mutableStateOf("") }
    var missingText by remember { mutableStateOf("") }
    var logText by remember { mutableStateOf("") }
    var logPathText by remember { mutableStateOf("") }
    var isRunning by remember { mutableStateOf(false) }
    var backendStatus by remember { mutableStateOf("Idle") }
    var backendReady by remember { mutableStateOf(false) }
    var backendModel by remember { mutableStateOf<ModelKind?>(null) }
    var backendExternalClip by remember { mutableStateOf(false) }
    var outputBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var envStatusText by remember { mutableStateOf("") }
    var envDetailText by remember { mutableStateOf("") }
    var downloadStatusText by remember { mutableStateOf("") }
    var downloadProgressText by remember { mutableStateOf("") }
    var downloadRunning by remember { mutableStateOf(false) }
    var openClAvailable by remember { mutableStateOf(false) }
    var sdxlAllowed by remember { mutableStateOf(true) }
    var sd15Allowed by remember { mutableStateOf(true) }
    val logFileRef = remember { AtomicReference<File?>(null) }
    val backendProcRef = remember { AtomicReference<Process?>(null) }
    val logBuffer = remember { StringBuilder() }
    val logLock = remember { Any() }
    val logDirty = remember { AtomicBoolean(false) }
    val hfCache = remember { mutableMapOf<String, List<HfFile>>() }
    val downloadCancel = remember { AtomicBoolean(false) }

    LaunchedEffect(Unit) {
        while (true) {
            delay(200L)
            if (!logDirty.getAndSet(false)) continue
            val snapshot = synchronized(logLock) {
                if (logBuffer.length > MAX_LOG_CHARS) {
                    logBuffer.delete(0, logBuffer.length - MAX_LOG_CHARS)
                }
                logBuffer.toString()
            }
            logText = snapshot
        }
    }

    fun appendLog(line: String) {
        val logFile = logFileRef.get()
        if (logFile != null) {
            try {
                logFile.appendText(line + "\n")
            } catch (_: Exception) {
            }
        }
        Log.i("SDXLCLML", line)
        synchronized(logLock) {
            logBuffer.append(line).append('\n')
        }
        logDirty.set(true)
        val m = Regex("\\[UNet\\] step (\\d+)/(\\d+)").find(line)
        if (m != null) {
            val next = "UNet ${m.groupValues[1]}/${m.groupValues[2]}"
            mainHandler.post { progressText = next }
        }
    }

    fun setStatus(line: String) {
        mainHandler.post { statusText = line }
    }

    fun setProgress(line: String) {
        mainHandler.post { progressText = line }
    }

    fun setOutput(line: String) {
        mainHandler.post { outputText = line }
    }

    fun setLogPath(line: String) {
        mainHandler.post { logPathText = line }
    }

    fun setMissing(line: String) {
        mainHandler.post { missingText = line }
    }

    fun setRunning(value: Boolean) {
        mainHandler.post { isRunning = value }
    }

    fun setBackendStatus(line: String) {
        mainHandler.post { backendStatus = line }
    }

    fun setBackendReady(value: Boolean) {
        mainHandler.post { backendReady = value }
    }

    fun setOutputBitmap(bitmap: Bitmap?) {
        mainHandler.post { outputBitmap = bitmap }
    }

    fun setDownloadStatus(line: String) {
        mainHandler.post { downloadStatusText = line }
    }

    fun setDownloadProgress(line: String) {
        mainHandler.post { downloadProgressText = line }
    }

    fun setDownloadRunning(value: Boolean) {
        mainHandler.post { downloadRunning = value }
    }

    fun updateEnvStatus() {
        val paths = buildPaths(context)
        val env = computeEnvStatus(context, paths.mnnFuseDir)
        val memText = String.format(Locale.US, "%.1f", env.totalMemGb)
        val envLine = "OpenCL/CLML: ${if (env.openClOk) "OK" else "Missing"} | RAM ${memText} GB"
        val modelLine = when {
            !env.openClOk -> "Available models: none (OpenCL/CLML missing)"
            env.sdxlOk -> "Available models: SD1.5, SDXL"
            else -> "Available models: SD1.5 only (SDXL needs >=${SDXL_MIN_MEM_GB}GB reported; device ${memText} GB)"
        }
        mainHandler.post {
            envStatusText = envLine
            envDetailText = modelLine
            openClAvailable = env.openClOk
            sdxlAllowed = env.sdxlOk
            sd15Allowed = env.openClOk
            if (modelKind == ModelKind.SDXL && !env.sdxlOk) {
                modelKind = ModelKind.SD15
                statusText = "SDXL disabled by RAM/driver check"
            }
        }
    }

    LaunchedEffect(Unit) {
        updateEnvStatus()
    }

    fun fetchHfFiles(prefix: String): List<HfFile> {
        hfCache[prefix]?.let { return it }
        val conn = URL(HF_API).openConnection() as HttpURLConnection
        conn.connectTimeout = 15000
        conn.readTimeout = 30000
        conn.setRequestProperty("User-Agent", "Fast-Diffusion")
        val payload = conn.inputStream.bufferedReader().use { it.readText() }
        val json = JSONObject(payload)
        val siblings = json.getJSONArray("siblings")
        val files = ArrayList<HfFile>(siblings.length())
        for (i in 0 until siblings.length()) {
            val item = siblings.getJSONObject(i)
            val path = item.getString("rfilename")
            if (!path.startsWith(prefix)) continue
            val lfs = item.optJSONObject("lfs")
            val size = lfs?.optLong("size", -1) ?: -1
            val oid = lfs?.optString("oid", "") ?: ""
            files.add(HfFile(path = path, size = size, sha256 = oid))
        }
        hfCache[prefix] = files
        return files
    }

    fun sha256Hex(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buf = ByteArray(1024 * 1024)
            while (true) {
                val read = input.read(buf)
                if (read <= 0) break
                digest.update(buf, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    fun isFileValid(hf: HfFile, baseDir: File): Boolean {
        val dest = File(baseDir, hf.path)
        if (!dest.exists()) return false
        if (hf.size > 0 && dest.length() != hf.size) return false
        if (hf.sha256.isNotEmpty()) {
            val existing = sha256Hex(dest)
            return existing.equals(hf.sha256, ignoreCase = true)
        }
        return true
    }

    fun downloadHfFile(hf: HfFile, baseDir: File): Boolean {
        if (downloadCancel.get()) return false
        val dest = File(baseDir, hf.path)
        dest.parentFile?.mkdirs()
        val part = File(dest.parentFile, dest.name + ".part")
        if (part.exists()) part.delete()
        if (dest.exists() && !isFileValid(hf, baseDir)) {
            dest.delete()
        }
        if (dest.exists()) {
            return true
        }

        val encoded = hf.path.split("/").joinToString("/") {
            URLEncoder.encode(it, "UTF-8").replace("+", "%20")
        }
        val url = URL(HF_RESOLVE + encoded)
        val conn = url.openConnection() as HttpURLConnection
        conn.connectTimeout = 15000
        conn.readTimeout = 30000
        conn.setRequestProperty("User-Agent", "Fast-Diffusion")
        if (conn.responseCode !in 200..299) {
            appendLog("Download failed: ${hf.path} code=${conn.responseCode}")
            return false
        }

        val tmp = File(dest.parentFile, dest.name + ".part")
        val digest = MessageDigest.getInstance("SHA-256")
        conn.inputStream.use { input ->
            FileOutputStream(tmp).use { output ->
                val buf = ByteArray(1024 * 1024)
                while (true) {
                    val read = input.read(buf)
                    if (read <= 0) break
                    if (downloadCancel.get()) {
                        tmp.delete()
                        return false
                    }
                    digest.update(buf, 0, read)
                    output.write(buf, 0, read)
                }
            }
        }
        val got = digest.digest().joinToString("") { "%02x".format(it) }
        if (hf.sha256.isNotEmpty() && !got.equals(hf.sha256, ignoreCase = true)) {
            tmp.delete()
            appendLog("SHA256 mismatch: ${hf.path}")
            return false
        }
        if (dest.exists()) dest.delete()
        if (!tmp.renameTo(dest)) {
            tmp.copyTo(dest, overwrite = true)
            tmp.delete()
        }
        return true
    }

    fun downloadMissingPrefix(prefix: String, baseDir: File): Boolean {
        setDownloadStatus("Fetching list: $prefix")
        val files = fetchHfFiles(prefix)
        if (files.isEmpty()) {
            setDownloadStatus("No files in HF: $prefix")
            return false
        }
        val missing = files.filterNot { isFileValid(it, baseDir) }
        if (missing.isEmpty()) {
            setDownloadStatus("All files already present")
            setDownloadProgress("")
            return true
        }
        val total = missing.size
        for ((idx, hf) in missing.sortedBy { it.path }.withIndex()) {
            if (downloadCancel.get()) {
                setDownloadStatus("Canceled")
                return false
            }
            setDownloadProgress("(${idx + 1}/$total) ${hf.path}")
            if (!downloadHfFile(hf, baseDir)) {
                if (downloadCancel.get()) {
                    setDownloadStatus("Canceled")
                    return false
                }
                setDownloadStatus("Failed: ${hf.path}")
                return false
            }
        }
        return true
    }

    fun checkResources() {
        updateEnvStatus()
        val paths = buildPaths(context)
        prepareRuntime(context, paths, ::appendLog)
        val env = computeEnvStatus(context, paths.mnnFuseDir)
        val memText = String.format(Locale.US, "%.1f", env.totalMemGb)
        val missing = mutableListOf<String>()
        if (!env.openClOk) {
            missing.add("OpenCL/CLML not available on this device")
        }
        if (modelKind == ModelKind.SDXL) {
            if (!env.sdxlOk) {
                missing.add("SDXL disabled: RAM ${memText} GB (<${SDXL_MIN_MEM_GB}GB reported) or OpenCL missing")
            }
            if (!paths.weightsDir.exists()) {
                missing.add("sdxl weights dir missing: ${paths.weightsDir}")
            } else {
                val weightsSub = File(paths.weightsDir, "weights")
                if (!weightsSub.exists()) {
                    missing.add("sdxl weights/ subdir missing: ${weightsSub}")
                }
            }
            val clipL = File(paths.mnnClipDir, "clip_int8.mnn")
            val clipG = File(paths.mnnClipDir, "clip_2_int8.mnn")
            val clipGW = File(paths.mnnClipDir, "clip_2_int8.mnn.weight")
            if (!clipL.exists()) missing.add("clip_int8.mnn missing: ${clipL}")
            if (!clipG.exists()) missing.add("clip_2_int8.mnn missing: ${clipG}")
            if (!clipGW.exists()) missing.add("clip_2_int8.mnn.weight missing: ${clipGW}")
            if (!paths.tokenizerJson.exists()) missing.add("tokenizer.json missing: ${paths.tokenizerJson}")
            if (!paths.tokenizer2Json.exists()) missing.add("tokenizer_2.json missing: ${paths.tokenizer2Json}")
            val backendBin = paths.backendBin
            val vaeBin = File(paths.binDir, "libsdxl_vae_decoder_run.so")
            val clipBin = paths.clipBin
            if (!backendBin.exists()) missing.add("sdxl backend bin missing: ${backendBin}")
            if (!vaeBin.exists()) missing.add("vae bin missing: ${vaeBin}")
            if (!clipBin.exists()) missing.add("clip bin missing: ${clipBin}")
        } else {
            if (!paths.sd15WeightsDir.exists()) {
                missing.add("sd15 weights dir missing: ${paths.sd15WeightsDir}")
            } else {
                val weightsSub = File(paths.sd15WeightsDir, "weights")
                if (!weightsSub.exists()) {
                    missing.add("sd15 weights/ subdir missing: ${weightsSub}")
                }
            }
            if (!paths.sd15TokenizerJson.exists()) missing.add("sd15 tokenizer.json missing: ${paths.sd15TokenizerJson}")
            val backendBin = paths.sd15BackendBin
            if (!backendBin.exists()) missing.add("sd15 backend bin missing: ${backendBin}")
        }
        val mnnLib = File(paths.mnnFuseDir, "libMNN.so")
        val mnnExpress = File(paths.mnnFuseDir, "libMNN_Express.so")
        val mnnCl = File(paths.mnnFuseDir, "libMNN_CL.so")
        val libcxx = File(paths.mnnFuseDir, "libc++_shared.so")
        if (!mnnLib.exists()) missing.add("libMNN.so missing: ${mnnLib}")
        if (!mnnExpress.exists()) missing.add("libMNN_Express.so missing: ${mnnExpress}")
        if (!mnnCl.exists()) missing.add("libMNN_CL.so missing: ${mnnCl}")
        if (!libcxx.exists()) missing.add("libc++_shared.so missing: ${libcxx}")
        if (missing.isEmpty()) {
            setMissing("Resources OK")
        } else {
            setMissing(missing.joinToString("\n"))
        }
    }

    fun cancelDownload() {
        if (!downloadRunning) return
        downloadCancel.set(true)
        setDownloadStatus("Canceling")
    }

    fun downloadModelAssets(kind: ModelKind) {
        if (downloadRunning) return
        downloadCancel.set(false)
        setDownloadRunning(true)
        setDownloadStatus("Starting")
        setDownloadProgress("")
        Thread {
            try {
                val paths = buildPaths(context)
                val ok = if (kind == ModelKind.SDXL) {
                    downloadMissingPrefix("MNN_clip/", paths.baseDir) &&
                        downloadMissingPrefix("sdxl_clml_weights/", paths.baseDir)
                } else {
                    downloadMissingPrefix("sd15_clml_weights/", paths.baseDir)
                }
                val canceled = downloadCancel.get()
                setDownloadStatus(
                    if (canceled) "Canceled" else if (ok) "Done" else "Failed"
                )
            } catch (e: Exception) {
                setDownloadStatus("Error: ${e.message}")
            } finally {
                setDownloadRunning(false)
                downloadCancel.set(false)
                updateEnvStatus()
                checkResources()
            }
        }.start()
    }

    fun waitForBackendReady(port: Int): Boolean {
        val deadline = System.currentTimeMillis() + BACKEND_HEALTH_TIMEOUT_MS
        while (System.currentTimeMillis() < deadline) {
            val proc = backendProcRef.get()
            if (proc == null || !proc.isAlive) {
                return false
            }
            try {
                val resp = httpGet("http://127.0.0.1:$port/health")
                if (resp.trim() == "ok") {
                    return true
                }
            } catch (e: Exception) {
                appendLog("Health check failed: ${e.message}")
            }
            Thread.sleep(500L)
        }
        return false
    }

    fun stopBackend() {
        val proc = backendProcRef.getAndSet(null)
        if (proc != null) {
            proc.destroy()
        }
        setBackendReady(false)
        setBackendStatus("Stopped")
        backendModel = null
        backendExternalClip = false
        setStatus("Backend stopped")
    }

    fun startBackend() {
        updateEnvStatus()
        if (!openClAvailable) {
            setStatus("OpenCL/CLML missing")
            setBackendStatus("Error")
            return
        }
        if (modelKind == ModelKind.SDXL && !sdxlAllowed) {
            setStatus("SDXL disabled by RAM/driver check")
            setBackendStatus("Error")
            return
        }
        val existing = backendProcRef.get()
        if (existing != null && existing.isAlive) {
            if (backendModel != modelKind) {
                appendLog("Backend model differs, restarting")
                stopBackend()
            } else {
                setBackendStatus("Running")
                setBackendReady(true)
                setStatus("Backend already running")
                return
            }
        }
        setBackendReady(false)
        setBackendStatus("Starting")
        statusText = "Preparing backend (1-2 min)"

        Thread {
            try {
                val paths = buildPaths(context)
                val logDir = File(paths.baseDir, "logs")
                logDir.mkdirs()
                val logFile = File(logDir, "backend_${System.currentTimeMillis()}.log")
                logFileRef.set(logFile)
                setLogPath(logFile.absolutePath)
                prepareRuntime(context, paths, ::appendLog)

                val missing = mutableListOf<String>()
                if (modelKind == ModelKind.SDXL) {
                    if (!paths.weightsDir.exists()) missing.add("sdxl weights dir missing")
                    if (!File(paths.weightsDir, "weights").exists()) missing.add("sdxl weights/ missing")
                    if (!File(paths.mnnClipDir, "clip_int8.mnn").exists()) missing.add("clip_int8.mnn missing")
                    if (!File(paths.mnnClipDir, "clip_2_int8.mnn").exists()) missing.add("clip_2_int8.mnn missing")
                    if (!File(paths.mnnClipDir, "clip_2_int8.mnn.weight").exists()) missing.add("clip_2_int8.mnn.weight missing")
                    if (!paths.tokenizerJson.exists()) missing.add("tokenizer.json missing")
                    if (!paths.tokenizer2Json.exists()) missing.add("tokenizer_2.json missing")
                    if (!paths.backendBin.exists()) missing.add("sdxl backend bin missing")
                    if (!paths.clipBin.exists()) missing.add("clip bin missing")
                    if (!File(paths.binDir, "libsdxl_vae_decoder_run.so").exists()) missing.add("vae bin missing")
                } else {
                    if (!paths.sd15WeightsDir.exists()) missing.add("sd15 weights dir missing")
                    if (!File(paths.sd15WeightsDir, "weights").exists()) missing.add("sd15 weights/ missing")
                    if (!paths.sd15BackendBin.exists()) missing.add("sd15 backend bin missing")
                    if (!paths.sd15TokenizerJson.exists()) missing.add("sd15 tokenizer.json missing")
                }
                if (!File(paths.mnnFuseDir, "libMNN.so").exists()) missing.add("libMNN.so missing")
                if (!File(paths.mnnFuseDir, "libMNN_Express.so").exists()) missing.add("libMNN_Express.so missing")
                if (!File(paths.mnnFuseDir, "libMNN_CL.so").exists()) missing.add("libMNN_CL.so missing")
                if (!File(paths.mnnFuseDir, "libc++_shared.so").exists()) missing.add("libc++_shared.so missing")
                if (missing.isNotEmpty()) {
                    setStatus("Missing resources")
                    setMissing(missing.joinToString("\n"))
                    setBackendStatus("Error")
                    return@Thread
                }

                val envCommon = mutableMapOf(
                    "LD_LIBRARY_PATH" to "${paths.mnnFuseDir}:/system/lib64:/vendor/lib64",
                    "CLML_NO_REUSE_TNN" to "1",
                )
                val port = if (modelKind == ModelKind.SDXL) BACKEND_PORT_SDXL else BACKEND_PORT_SD15
                val cmd = if (modelKind == ModelKind.SDXL) {
                    if (useExternalClip) {
                        envCommon["SDXL_BACKEND_SKIP_CLIP"] = "1"
                        envCommon["SDXL_BACKEND_LAZY_UNET"] = "1"
                        envCommon["SDXL_BACKEND_RELEASE_UNET"] = "1"
                    } else {
                        envCommon["SDXL_BACKEND_LAZY_CLIP"] = "1"
                        envCommon["SDXL_BACKEND_RELEASE_CLIP"] = "1"
                        envCommon["SDXL_BACKEND_LAZY_UNET"] = "1"
                        envCommon["SDXL_BACKEND_RELEASE_UNET"] = "1"
                    }
                    listOf(
                        paths.backendBin.absolutePath,
                        paths.weightsDir.absolutePath,
                        paths.mnnClipDir.absolutePath,
                        paths.tokenizerJson.absolutePath,
                        paths.tokenizer2Json.absolutePath,
                        paths.workDir.absolutePath,
                        port.toString(),
                        SDXL_HEIGHT.toString(),
                        SDXL_WIDTH.toString()
                    )
                } else {
                    envCommon["CLML_KEEP_TEXT_ENCODER"] = "1"
                    listOf(
                        paths.sd15BackendBin.absolutePath,
                        paths.sd15WeightsDir.absolutePath,
                        paths.workDir.absolutePath,
                        port.toString(),
                        paths.sd15TokenizerJson.absolutePath
                    )
                }

                val pb = ProcessBuilder(cmd)
                pb.directory(paths.workDir)
                pb.redirectErrorStream(true)
                pb.environment().putAll(envCommon)
                val proc = pb.start()
                backendProcRef.set(proc)
                backendModel = modelKind
                backendExternalClip = (modelKind == ModelKind.SDXL) && useExternalClip

                Thread {
                    val reader = BufferedReader(InputStreamReader(proc.inputStream))
                    while (true) {
                        val line = reader.readLine() ?: break
                        appendLog(line)
                    }
                }.start()

                Thread {
                    val exit = proc.waitFor()
                    appendLog("Backend exit code: $exit")
                    setBackendReady(false)
                    setBackendStatus("Exited")
                }.start()

                setStatus("Waiting for backend")
                val ok = waitForBackendReady(port)
                if (ok) {
                    setBackendReady(true)
                    setBackendStatus("Running")
                    setStatus("Backend ready")
                } else {
                    setBackendStatus("Error")
                    setStatus("Backend health failed")
                }
            } catch (e: Exception) {
                appendLog("Backend error: ${e.message}")
                setBackendStatus("Error")
                setStatus("Backend start failed")
            }
        }.start()
    }

    fun runGenerate() {
        if (isRunning) return
        if (!backendReady) {
            setStatus("Backend not ready")
            return
        }
        setRunning(true)
        setOutputBitmap(null)
        progressText = ""
        statusText = "Generating"

        Thread {
            try {
                val steps = max(1, stepsText.toIntOrNull() ?: 20)
                val cfg = cfgText.toFloatOrNull() ?: 7.5f
                val seed = seedText.toIntOrNull() ?: 0
                val earlyK = max(0, earlyKText.toIntOrNull() ?: 0)

                val paths = buildPaths(context)
                prepareRuntime(context, paths, ::appendLog)
                val currentModel = modelKind
                if (backendReady && backendModel != currentModel) {
                    appendLog("Backend model mismatch: restarting")
                    stopBackend()
                    setStatus("Backend stopped, please init")
                    setRunning(false)
                    return@Thread
                }
                if (currentModel == ModelKind.SDXL && backendExternalClip != useExternalClip) {
                    appendLog("Backend config mismatch: restarting")
                    stopBackend()
                    setStatus("Backend stopped, please init")
                    setRunning(false)
                    return@Thread
                }

                if (currentModel == ModelKind.SDXL) {
                    val request = linkedMapOf(
                        "prompt" to promptText,
                        "negative_prompt" to negativeText,
                        "steps" to steps.toString(),
                        "cfg" to cfg.toString(),
                        "seed" to seed.toString(),
                        "height" to SDXL_HEIGHT.toString(),
                        "width" to SDXL_WIDTH.toString(),
                        "early_k" to earlyK.toString(),
                        "decode_x0" to if (decodeX0) "1" else "0"
                    )

                    if (useExternalClip) {
                        val clipTag = System.currentTimeMillis().toString()
                        val textOut = File(paths.workDir, "clip_text_${clipTag}.bin")
                        val pooledOut = File(paths.workDir, "clip_pooled_${clipTag}.bin")
                        val clipCmd = listOf(
                            paths.clipBin.absolutePath,
                            paths.mnnClipDir.absolutePath,
                            paths.tokenizerJson.absolutePath,
                            paths.tokenizer2Json.absolutePath,
                            promptText,
                            negativeText,
                            textOut.absolutePath,
                            pooledOut.absolutePath,
                        )
                        val clipEnv = mapOf(
                            "LD_LIBRARY_PATH" to "${paths.mnnFuseDir}:/system/lib64:/vendor/lib64",
                        )
                        setStatus("Running CLIP")
                        setProgress("CLIP")
                        val clipExit = runProcess(clipCmd, clipEnv, paths.workDir) { line ->
                            appendLog(line)
                        }
                        if (clipExit != 0) {
                            setStatus("CLIP failed: code=$clipExit")
                            setRunning(false)
                            return@Thread
                        }
                        if (!textOut.exists() || !pooledOut.exists()) {
                            setStatus("CLIP output missing")
                            setRunning(false)
                            return@Thread
                        }
                        request["text_embed_path"] = textOut.absolutePath
                        request["pooled_embed_path"] = pooledOut.absolutePath
                    }

                    setStatus("Waiting backend response")
                    val resp = httpPostForm("http://127.0.0.1:$BACKEND_PORT_SDXL/generate", request)
                    appendLog("Backend response: $resp")
                    val json = JSONObject(resp)
                    if (json.has("error")) {
                        setStatus("Backend error")
                        appendLog("Backend error: ${json.getString("error")}")
                        setRunning(false)
                        return@Thread
                    }
                    val latentPath = json.getString("latent_path")
                    appendLog("Latent path: $latentPath")
                    if (!waitForFile(File(latentPath), 60000)) {
                        setStatus("Latent missing")
                        appendLog("Latent not found after wait")
                        setRunning(false)
                        return@Thread
                    }
                    val localTmpVaeDir = File("/data/local/tmp/sdxl_app_vae")
                    val vaeWorkDir = if (localTmpVaeDir.exists() && localTmpVaeDir.canWrite()) {
                        localTmpVaeDir
                    } else {
                        paths.workDir
                    }
                    val localTmpMnnDir = File("/data/local/tmp/MNN_fuse")
                    val mnnDirForVae = if (localTmpMnnDir.exists() && localTmpMnnDir.canRead()) {
                        localTmpMnnDir
                    } else {
                        paths.mnnFuseDir
                    }
                    val useLocalTmpMnn = mnnDirForVae == localTmpMnnDir
                    val vaeOut = File(vaeWorkDir, "output/sdxl_vae_out.qfp32")
                    appendLog("VAE workDir: ${vaeWorkDir.absolutePath}")
                    appendLog("VAE MNN dir: ${mnnDirForVae.absolutePath}")

                    val envCommon = mapOf(
                        "LD_LIBRARY_PATH" to "${mnnDirForVae.absolutePath}:/system/lib64:/vendor/lib64",
                        "MNN_CL_LIB" to File(mnnDirForVae, "libMNN_CL.so").absolutePath,
                        "MNN_BACKEND" to "opencl",
                        "MNN_GPU_MODE" to "1",
                        "MNN_MEM" to "0",
                        "MNN_POWER" to "0",
                        "MNN_PREC" to "0",
                        "CLML_NO_REUSE_TNN" to "1",
                        "CLML_MNN_ATTN_BACKEND" to "opencl",
                        "CLML_MNN_ATTN_FP32" to "1",
                    )

                    val vaeCmd = listOf(
                        File(paths.binDir, "libsdxl_vae_decoder_run.so").absolutePath,
                        paths.weightsDir.absolutePath,
                        "1",
                        "0",
                        "1",
                        "0.1",
                        "128",
                        "128",
                        "1",
                        latentPath
                    )

                    setStatus("Running VAE")
                    setProgress("VAE")
                    var vaeExit = runProcess(vaeCmd, envCommon, vaeWorkDir) { line ->
                        appendLog(line)
                    }
                    if (vaeExit != 0 && useLocalTmpMnn) {
                        appendLog("VAE retry with app libs")
                        val envFallback = envCommon.toMutableMap()
                        envFallback["LD_LIBRARY_PATH"] = "${paths.mnnFuseDir.absolutePath}:/system/lib64:/vendor/lib64"
                        envFallback["MNN_CL_LIB"] = File(paths.mnnFuseDir, "libMNN_CL.so").absolutePath
                        vaeExit = runProcess(vaeCmd, envFallback, vaeWorkDir) { line ->
                            appendLog(line)
                        }
                    }
                    if (vaeExit != 0) {
                        setStatus("VAE failed: code=$vaeExit")
                        setRunning(false)
                        return@Thread
                    }
                    if (!vaeOut.exists()) {
                        setStatus("VAE output missing")
                        setRunning(false)
                        return@Thread
                    }

                    val timeTag = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                    val outBase = if (paths.externalDir.exists() || paths.externalDir.mkdirs()) {
                        paths.externalDir
                    } else {
                        paths.baseDir
                    }
                    val outDir = File(outBase, "outputs")
                    outDir.mkdirs()
                    val outQfp32 = File(outDir, "sdxl_${timeTag}.qfp32")
                    val outPng = File(outDir, "sdxl_${timeTag}.png")
                    vaeOut.copyTo(outQfp32, overwrite = true)

                    setStatus("Converting to PNG")
                    decodeQfp32ToPng(vaeOut, outPng, SDXL_WIDTH, SDXL_HEIGHT)

                    setStatus("Done")
                    setProgress("Done")
                    setOutput(outPng.absolutePath)
                    setOutputBitmap(BitmapFactory.decodeFile(outPng.absolutePath))
                } else {
                    setStatus("Running SD1.5")
                    setProgress("SD1.5")
                    val request = linkedMapOf(
                        "prompt" to promptText,
                        "negative_prompt" to negativeText,
                        "steps" to steps.toString(),
                        "cfg" to cfg.toString(),
                        "seed" to seed.toString(),
                    )
                    val resp = httpPostForm("http://127.0.0.1:$BACKEND_PORT_SD15/generate", request)
                    appendLog("Backend response: $resp")
                    val json = JSONObject(resp)
                    if (json.has("error")) {
                        setStatus("Backend error")
                        appendLog("Backend error: ${json.getString("error")}")
                        setRunning(false)
                        return@Thread
                    }
                    val outPath = json.getString("output_path")
                    appendLog("Output path: $outPath")
                    if (!waitForFile(File(outPath), 60000)) {
                        setStatus("Output missing")
                        appendLog("Output not found after wait")
                        setRunning(false)
                        return@Thread
                    }

                    val timeTag = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                    val outBase = if (paths.externalDir.exists() || paths.externalDir.mkdirs()) {
                        paths.externalDir
                    } else {
                        paths.baseDir
                    }
                    val outDir = File(outBase, "outputs")
                    outDir.mkdirs()
                    val outQfp32 = File(outDir, "sd15_${timeTag}.qfp32")
                    val outPng = File(outDir, "sd15_${timeTag}.png")
                    File(outPath).copyTo(outQfp32, overwrite = true)

                    setStatus("Converting to PNG")
                    decodeQfp32ToPng(File(outPath), outPng, SD15_WIDTH, SD15_HEIGHT)

                    setStatus("Done")
                    setProgress("Done")
                    setOutput(outPng.absolutePath)
                    setOutputBitmap(BitmapFactory.decodeFile(outPng.absolutePath))
                }
            } catch (e: Exception) {
                appendLog("Error: ${e.message}")
                setStatus("Error")
            } finally {
                setRunning(false)
            }
        }.start()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("SDXL/SD1.5 CLML", style = MaterialTheme.typography.titleLarge)

        fun switchModel(next: ModelKind) {
            if (modelKind == next) return
            if (next == ModelKind.SDXL && !sdxlAllowed) {
                statusText = "SDXL disabled by RAM/driver check"
                return
            }
            if (next == ModelKind.SD15 && !sd15Allowed) {
                statusText = "SD1.5 disabled (OpenCL/CLML missing)"
                return
            }
            if (backendReady || (backendProcRef.get()?.isAlive == true)) {
                appendLog("Stopping backend for model switch")
                stopBackend()
            }
            modelKind = next
        }

        Row(verticalAlignment = Alignment.CenterVertically) {
            RadioButton(
                selected = modelKind == ModelKind.SDXL,
                onClick = { switchModel(ModelKind.SDXL) },
                enabled = sdxlAllowed,
            )
            Text("SDXL 1024")
            Spacer(modifier = Modifier.width(8.dp))
            RadioButton(
                selected = modelKind == ModelKind.SD15,
                onClick = { switchModel(ModelKind.SD15) },
                enabled = sd15Allowed,
            )
            Text("SD1.5 512 (UNet resident)")
        }

        OutlinedTextField(
            value = promptText,
            onValueChange = { promptText = it },
            label = { Text("Prompt") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 3,
        )
        OutlinedTextField(
            value = negativeText,
            onValueChange = { negativeText = it },
            label = { Text("Negative Prompt") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 2,
        )

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(
                value = stepsText,
                onValueChange = { stepsText = it },
                label = { Text("Steps") },
                modifier = Modifier.weight(1f),
            )
            OutlinedTextField(
                value = cfgText,
                onValueChange = { cfgText = it },
                label = { Text("CFG") },
                modifier = Modifier.weight(1f),
            )
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(
                value = seedText,
                onValueChange = { seedText = it },
                label = { Text("Seed") },
                modifier = Modifier.weight(1f),
            )
            if (modelKind == ModelKind.SDXL) {
                OutlinedTextField(
                    value = earlyKText,
                    onValueChange = { earlyKText = it },
                    label = { Text("Early K") },
                    modifier = Modifier.weight(1f),
                )
            } else {
                Spacer(modifier = Modifier.weight(1f))
            }
        }
        if (modelKind == ModelKind.SDXL) {

            Row(verticalAlignment = Alignment.CenterVertically) {
                Checkbox(checked = decodeX0, onCheckedChange = { decodeX0 = it })
                Text("Decode x0")
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Checkbox(checked = useExternalClip, onCheckedChange = { useExternalClip = it })
                Text("External CLIP process")
            }
        }

        Text("Environment")
        if (envStatusText.isNotEmpty()) {
            Text(envStatusText)
        }
        if (envDetailText.isNotEmpty()) {
            Text(envDetailText)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = { checkResources() },
                enabled = !isRunning,
                modifier = Modifier.weight(1f),
            ) {
                Text("Check resources")
            }
            Button(
                onClick = { startBackend() },
                enabled = !isRunning,
                modifier = Modifier.weight(1f),
            ) {
                Text("Init backend")
            }
            Button(
                onClick = { stopBackend() },
                enabled = !isRunning,
                modifier = Modifier.weight(1f),
            ) {
                Text("Stop")
            }
        }
        Button(
            onClick = { runGenerate() },
            enabled = !isRunning && backendReady,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Generate")
        }

        Text("Downloads")
        Button(
            onClick = { downloadModelAssets(ModelKind.SDXL) },
            enabled = !downloadRunning,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Download SDXL (missing only)")
        }
        Button(
            onClick = { downloadModelAssets(ModelKind.SD15) },
            enabled = !downloadRunning,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Download SD1.5 (missing only)")
        }
        Button(
            onClick = { cancelDownload() },
            enabled = downloadRunning,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Stop download")
        }
        if (downloadStatusText.isNotEmpty()) {
            Text("Download: $downloadStatusText")
        }
        if (downloadProgressText.isNotEmpty()) {
            Text(downloadProgressText)
        }

        val paths = buildPaths(context)
        Text("Base dir: ${paths.baseDir}")
        Text("Output dir: ${paths.externalDir}")
        if (modelKind == ModelKind.SDXL) {
            Text("Weights dir: ${paths.weightsDir}")
            Text("MNN clip dir: ${paths.mnnClipDir}")
            Text("Tokenizer: ${paths.tokenizerJson}")
        } else {
            Text("Weights dir: ${paths.sd15WeightsDir}")
            Text("Tokenizer: ${paths.sd15TokenizerJson}")
        }
        Text("Tokenizer2: ${paths.tokenizer2Json}")
        Text("Backend: ${if (modelKind == ModelKind.SDXL) paths.backendBin else paths.sd15BackendBin}")
        if (logPathText.isNotEmpty()) {
            Text("Log file: $logPathText")
        }

        if (missingText.isNotEmpty()) {
            Text("Resources: $missingText")
        }
        Text("Status: $statusText")
        Text("Backend: $backendStatus")
        if (progressText.isNotEmpty()) {
            Text("Progress: $progressText")
        }
        if (outputText.isNotEmpty()) {
            Text("Output: $outputText")
        }

        if (outputBitmap != null) {
            Text("Preview:")
            Image(
                bitmap = outputBitmap!!.asImageBitmap(),
                contentDescription = "generated image",
                modifier = Modifier.fillMaxWidth(),
                contentScale = ContentScale.FillWidth,
            )
        }

        Spacer(modifier = Modifier.height(8.dp))
        Text("Logs:")
        SelectionContainer {
            Text(logText, fontFamily = FontFamily.Monospace)
        }
    }
}

private fun buildPaths(context: android.content.Context): RuntimePaths {
    val runtimeDir = File(context.filesDir, "runtime")
    val nativeDir = File(context.applicationInfo.nativeLibraryDir)
    val binDir = nativeDir
    val mnnFuseDir = nativeDir
    val workDir = File(runtimeDir, "work")

    val baseDir = context.filesDir
    val extDir = context.getExternalFilesDir(null) ?: baseDir
    val weightsDir = File(baseDir, "sdxl_clml_weights")
    val sd15WeightsDir = File(baseDir, "sd15_clml_weights")
    val mnnClipDir = File(baseDir, "MNN_clip")
    val tokensDir = File(baseDir, "tokens")
    val clipL = File(tokensDir, "clip_l_ids.i32")
    val clipG = File(tokensDir, "clip_g_ids.i32")
    val tokenizerJson = File(tokensDir, "tokenizer.json")
    val tokenizer2Json = File(tokensDir, "tokenizer_2.json")
    val sd15TokenizerJson = File(tokensDir, "tokenizer_sd15.json")
    val backendBin = File(binDir, "libsdxl_clml_backend.so")
    val sd15BackendBin = File(binDir, "libsd15_clml_backend.so")
    val clipBin = File(binDir, "libsdxl_clip_run.so")

    workDir.mkdirs()
    File(workDir, "output").mkdirs()
    tokensDir.mkdirs()

    return RuntimePaths(
        runtimeDir = runtimeDir,
        binDir = binDir,
        mnnFuseDir = mnnFuseDir,
        workDir = workDir,
        baseDir = baseDir,
        externalDir = extDir,
        weightsDir = weightsDir,
        sd15WeightsDir = sd15WeightsDir,
        mnnClipDir = mnnClipDir,
        tokensDir = tokensDir,
        clipL = clipL,
        clipG = clipG,
        tokenizerJson = tokenizerJson,
        tokenizer2Json = tokenizer2Json,
        sd15TokenizerJson = sd15TokenizerJson,
        backendBin = backendBin,
        sd15BackendBin = sd15BackendBin,
        clipBin = clipBin,
    )
}

private fun prepareRuntime(
    context: android.content.Context,
    paths: RuntimePaths,
    log: (String) -> Unit,
) {
    paths.runtimeDir.mkdirs()

    copyAssetIfMissing(context, "tokens/clip_l_ids.i32", paths.clipL, false, log)
    copyAssetIfMissing(context, "tokens/clip_g_ids.i32", paths.clipG, false, log)
    copyAssetIfMissing(context, "tokenizers/tokenizer.json", paths.tokenizerJson, false, log)
    copyAssetIfMissing(context, "tokenizers/tokenizer_2.json", paths.tokenizer2Json, false, log)
    copyAssetIfMissing(context, "tokenizers/sd15_tokenizer.json", paths.sd15TokenizerJson, false, log)
}

private fun copyAssetIfMissing(
    context: android.content.Context,
    assetPath: String,
    outFile: File,
    executable: Boolean,
    log: (String) -> Unit,
) {
    if (outFile.exists()) {
        if (executable) outFile.setExecutable(true)
        return
    }
    outFile.parentFile?.mkdirs()
    context.assets.open(assetPath).use { input ->
        FileOutputStream(outFile).use { output ->
            val buf = ByteArray(16 * 1024)
            while (true) {
                val read = input.read(buf)
                if (read <= 0) break
                output.write(buf, 0, read)
            }
        }
    }
    if (executable) outFile.setExecutable(true)
    log("Extracted asset: $assetPath -> ${outFile.absolutePath}")
}

private fun runProcess(
    command: List<String>,
    env: Map<String, String>,
    workDir: File,
    onLine: (String) -> Unit,
): Int {
    val pb = ProcessBuilder(command)
    pb.directory(workDir)
    pb.redirectErrorStream(true)
    pb.environment().putAll(env)
    val proc = pb.start()

    val reader = BufferedReader(InputStreamReader(proc.inputStream))
    while (true) {
        val line = reader.readLine() ?: break
        onLine(line)
    }
    return proc.waitFor()
}

private fun httpGet(url: String): String {
    val conn = URL(url).openConnection() as HttpURLConnection
    conn.connectTimeout = 1000
    conn.readTimeout = 1000
    conn.requestMethod = "GET"
    val stream = if (conn.responseCode in 200..299) conn.inputStream else conn.errorStream
    return stream.bufferedReader().use { it.readText() }
}

private fun httpPostForm(url: String, params: Map<String, String>): String {
    val body = params.entries.joinToString("&") { (k, v) ->
        "${URLEncoder.encode(k, "UTF-8")}=${URLEncoder.encode(v, "UTF-8")}"
    }
    val conn = URL(url).openConnection() as HttpURLConnection
    conn.connectTimeout = 10000
    conn.readTimeout = BACKEND_GENERATE_TIMEOUT_MS
    conn.requestMethod = "POST"
    conn.doOutput = true
    conn.setRequestProperty("Content-Type", "application/x-www-form-urlencoded")
    conn.outputStream.use { it.write(body.toByteArray(Charsets.UTF_8)) }
    val stream = if (conn.responseCode in 200..299) conn.inputStream else conn.errorStream
    return stream.bufferedReader().use { it.readText() }
}

private fun waitForFile(file: File, timeoutMs: Long): Boolean {
    val deadline = System.currentTimeMillis() + timeoutMs
    while (System.currentTimeMillis() < deadline) {
        if (file.exists()) return true
        Thread.sleep(200L)
    }
    return file.exists()
}

private fun decodeQfp32ToPng(input: File, output: File, width: Int, height: Int) {
    val count = width * height
    val floats = FloatArray(count * 3)
    FileInputStream(input).use { fis ->
        val buffer = ByteArray(floats.size * 4)
        var total = 0
        while (total < buffer.size) {
            val read = fis.read(buffer, total, buffer.size - total)
            if (read <= 0) break
            total += read
        }
        val bb = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN)
        bb.asFloatBuffer().get(floats)
    }

    val pixels = IntArray(count)
    val hw = width * height
    for (i in 0 until count) {
        val r = clampToByte((floats[i] * 0.5f + 0.5f) * 255.0f)
        val g = clampToByte((floats[i + hw] * 0.5f + 0.5f) * 255.0f)
        val b = clampToByte((floats[i + hw * 2] * 0.5f + 0.5f) * 255.0f)
        pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }

    val bitmap = android.graphics.Bitmap.createBitmap(width, height, android.graphics.Bitmap.Config.ARGB_8888)
    bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
    FileOutputStream(output).use { fos ->
        bitmap.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, fos)
    }
}

private fun clampToByte(value: Float): Int {
    val v = min(255.0f, max(0.0f, value))
    return (v + 0.5f).toInt()
}
