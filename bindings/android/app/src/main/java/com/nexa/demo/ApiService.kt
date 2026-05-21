package ai.nexa.demo

import ai.nexa.core.*
import android.app.*
import android.content.Intent
import android.os.IBinder
import androidx.core.app.NotificationCompat
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.coroutines.*
import kotlinx.serialization.Serializable

// --- Request/Response shapes (OpenAI-compatible) ---

@Serializable
data class ChatMsg(val role: String, val content: String)

@Serializable
data class ChatRequest(
    val messages: List<ChatMsg>,
    val max_tokens: Int = 512,
    val stream: Boolean = false   // streaming support can be added later
)

@Serializable
data class ChatChoice(val index: Int, val message: ChatMsg)

@Serializable
data class ChatResponse(val choices: List<ChatChoice>)

// --- The Service ---

class ApiService : Service() {

    private var llmWrapper: LlmWrapper? = null
    private var server: ApplicationEngine? = null
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    companion object {
        const val EXTRA_MODEL_PATH = "model_path"
        const val EXTRA_DEVICE = "device"      // "cpu", "gpu", "npu"
        const val PORT = 8080
        const val NOTIF_CHANNEL = "llm_api"
        const val NOTIF_ID = 1
	const val EXTRA_CTX_LENGTH = "ctx_length"
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val modelPath = intent?.getStringExtra(EXTRA_MODEL_PATH) ?: return START_NOT_STICKY
        val device = intent.getStringExtra(EXTRA_DEVICE) ?: "cpu"

        startForeground(NOTIF_ID, buildNotification("Loading model..."))
        scope.launch { loadModel(modelPath, device) }

        return START_STICKY
    }

    private suspend fun loadModel(modelPath: String, device: String) {
        // Map device string → Nexa SDK params
        // "cpu"  → nGpuLayers=0,  device_id=null
        // "gpu"  → nGpuLayers=999, device_id="gpu"
        // "npu"  → nGpuLayers=999, device_id="dev0"  (GGML Hexagon backend)
        val (gpuLayers, deviceId) = when (device) {
            "gpu" -> Pair(999, "gpu")
            "npu" -> Pair(999, "dev0")
            else  -> Pair(0, null)
        }
	val ctxLen = intent?.getIntExtra(EXTRA_CTX_LENGTH, 4096) ?: 4096


        LlmWrapper.builder()
            .llmCreateInput(
                LlmCreateInput(
                    model_name = "",           // empty for GGUF
                    model_path = modelPath,
                    config = ModelConfig(
                        nCtx = ctxLen,
                        nGpuLayers = gpuLayers
                    ),
                    plugin_id = "cpu_gpu",     // handles cpu/gpu/hexagon-npu
                    device_id = deviceId
                )
            )
            .build()
            .onSuccess { wrapper ->
                llmWrapper = wrapper
                updateNotification("API running on localhost:$PORT ($device)")
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
                // Health check
                get("/health") {
                    call.respondText("ok")
                }

                // OpenAI-compatible chat endpoint
                post("/v1/chat/completions") {
                    val wrapper = llmWrapper
                    if (wrapper == null) {
                        call.respond(io.ktor.http.HttpStatusCode.ServiceUnavailable,
                            mapOf("error" to "Model not loaded"))
                        return@post
                    }

                    val req = call.receive<ChatRequest>()

                    // Apply chat template (handles system/user/assistant turns)
                    val chatArray = req.messages
                        .map { ChatMessage(it.role, it.content) }
                        .toTypedArray()

                    wrapper.applyChatTemplate(chatArray, null, false)
                        .onSuccess { template ->
                            val sb = StringBuilder()
                            val genConfig = GenerationConfig(maxTokens = req.max_tokens)

                            wrapper.generateStreamFlow(template.formattedText, genConfig)
                                .collect { result ->
                                    when (result) {
                                        is LlmStreamResult.Token     -> sb.append(result.text)
                                        is LlmStreamResult.Completed -> { /* done */ }
                                        is LlmStreamResult.Error     ->
                                            sb.append("[error: ${result.throwable.message}]")
                                    }
                                }

                            call.respond(ChatResponse(
                                choices = listOf(ChatChoice(
                                    index = 0,
                                    message = ChatMsg("assistant", sb.toString())
                                ))
                            ))
                        }
                        .onFailure { err ->
                            call.respond(io.ktor.http.HttpStatusCode.InternalServerError,
                                mapOf("error" to err.message))
                        }
                }
            }
        }.start(wait = false)
    }

    // --- Foreground notification boilerplate ---

    private fun buildNotification(text: String): Notification {
        val chan = NotificationChannel(
            NOTIF_CHANNEL, "LLM API Service",
            NotificationManager.IMPORTANCE_LOW
        )
        getSystemService(NotificationManager::class.java)
            .createNotificationChannel(chan)

        return NotificationCompat.Builder(this, NOTIF_CHANNEL)
            .setContentTitle("Nexa LLM API")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .build()
    }

    private fun updateNotification(text: String) {
        val nm = getSystemService(NotificationManager::class.java)
        nm.notify(NOTIF_ID, buildNotification(text))
    }

    override fun onDestroy() {
        server?.stop(500, 500)
        llmWrapper?.close()
        scope.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
