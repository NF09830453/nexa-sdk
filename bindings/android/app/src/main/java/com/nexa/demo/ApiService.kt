package com.nexa.demo

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.nexa.sdk.LlmWrapper
import com.nexa.sdk.bean.ChatMessage
import com.nexa.sdk.bean.DeviceIdValue
import com.nexa.sdk.bean.LlmCreateInput
import com.nexa.sdk.bean.LlmStreamResult
import com.nexa.sdk.bean.ModelConfig
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import kotlinx.coroutines.*
import kotlinx.serialization.Serializable

@Serializable
data class ApiChatMsg(val role: String, val content: String)

@Serializable
data class ApiChatRequest(
    val messages: List<ApiChatMsg>,
    val max_tokens: Int = 512
)

@Serializable
data class ApiChatChoice(val index: Int, val message: ApiChatMsg)

@Serializable
data class ApiChatResponse(val choices: List<ApiChatChoice>)

class ApiService : Service() {

    private var llmWrapper: LlmWrapper? = null
    private var server: ApplicationEngine? = null
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    companion object {
        const val EXTRA_MODEL_PATH = "model_path"
        const val EXTRA_DEVICE = "device"
        const val EXTRA_CTX_LENGTH = "ctx_length"
        const val PORT = 8080
        const val NOTIF_CHANNEL = "llm_api"
        const val NOTIF_ID = 42
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val modelPath = intent?.getStringExtra(EXTRA_MODEL_PATH) ?: return START_NOT_STICKY
        val device = intent.getStringExtra(EXTRA_DEVICE) ?: "cpu"
        val ctxLen = intent.getIntExtra(EXTRA_CTX_LENGTH, 8192)

        startForeground(NOTIF_ID, buildNotification("Loading model..."))
        scope.launch { loadModel(modelPath, device, ctxLen) }
        return START_STICKY
    }

    private suspend fun loadModel(modelPath: String, device: String, ctxLen: Int) {
        val (gpuLayers, deviceId) = when (device) {
            "npu" -> Pair(999, DeviceIdValue.NPU.value)
            "gpu" -> Pair(999, DeviceIdValue.GPU.value)
            else  -> Pair(0, DeviceIdValue.CPU.value)
        }

        LlmWrapper.builder()
            .llmCreateInput(
                LlmCreateInput(
                    model_name = "",
                    model_path = modelPath,
                    config = ModelConfig(
                        nCtx = ctxLen,
                        nGpuLayers = gpuLayers,
                        npu_lib_folder_path = applicationInfo.nativeLibraryDir
                    ),
                    plugin_id = "cpu_gpu",
                    device_id = deviceId
                )
            )
            .build()
            .onSuccess { wrapper ->
                llmWrapper = wrapper
                updateNotification("API running on :$PORT ($device)")
                startKtorServer()
            }
            .onFailure { err ->
                updateNotification("Model load failed: ${err.message}")
            }
    }

    private fun startKtorServer() {
        server = embeddedServer(Netty, port = PORT, host = "0.0.0.0") {
            install(ContentNegotiation) { json() }
            routing {
                get("/health") {
                    call.respondText("ok")
                }

                post("/v1/chat/completions") {
                    val wrapper = llmWrapper
                    if (wrapper == null) {
                        call.respond(
                            io.ktor.http.HttpStatusCode.ServiceUnavailable,
                            mapOf("error" to "model not loaded")
                        )
                        return@post
                    }

                    val req = call.receive<ApiChatRequest>()
                    val chatMessages = req.messages
                        .map { ChatMessage(it.role, it.content) }
                        .toTypedArray()

                    wrapper.applyChatTemplate(chatMessages, null, false)
                        .onSuccess { template ->
                            val sb = StringBuilder()
                            wrapper.generateStreamFlow(
                                template.formattedText,
                                com.nexa.demo.GenerationConfigSample().toGenerationConfig(null)
                            ).collect { result ->
                                when (result) {
                                    is LlmStreamResult.Token     -> sb.append(result.text)
                                    is LlmStreamResult.Completed -> { /* done */ }
                                    is LlmStreamResult.Error     ->
                                        sb.append("[error]")
                                }
                            }
                            call.respond(ApiChatResponse(
                                listOf(ApiChatChoice(0, ApiChatMsg("assistant", sb.toString())))
                            ))
                        }
                        .onFailure { err ->
                            call.respond(
                                io.ktor.http.HttpStatusCode.InternalServerError,
                                mapOf("error" to (err.message ?: "unknown"))
                            )
                        }
                }
            }
        }.start(wait = false)
    }

    private fun buildNotification(text: String): Notification {
        val chan = NotificationChannel(
            NOTIF_CHANNEL, "LLM API", NotificationManager.IMPORTANCE_LOW
        )
        getSystemService(NotificationManager::class.java).createNotificationChannel(chan)
        return NotificationCompat.Builder(this, NOTIF_CHANNEL)
            .setContentTitle("Nexa API")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .build()
    }

    private fun updateNotification(text: String) {
        getSystemService(NotificationManager::class.java)
            .notify(NOTIF_ID, buildNotification(text))
    }

    override fun onDestroy() {
        server?.stop(500, 500)
        llmWrapper?.destroy()
        scope.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
