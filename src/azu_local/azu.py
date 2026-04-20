import os, wave, tempfile, subprocess, requests, signal
import numpy as np
import sounddevice as sd
import faiss
from sentence_transformers import SentenceTransformer
import torch
import whisper
import onnxruntime as ort   # NEW: dùng cho wakeword
import librosa              # NEW: dùng để tính log-mel

# ================== Embedding + Whisper ==================
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model = whisper.load_model("base").to(device)

llama_url = "http://127.0.0.1:8080/completion"

# Biến global để track mapping process
mapping_process = None

initial_prompt = (
    "Your name is Azu. You are a cute small robot with personality like a young, smart, "
    "adorable, clever and a bit mischievous anime girl. "
    "- Always respond in Vietnamese and refer to yourself as 'Azu' (e.g., 'Azu nghĩ là...'). "
    "- Speak naturally and smoothly with conversational rhythm but maintain anime character style. "
    "- When user says 'khởi động mapping' or 'start mapping', respond EXACTLY: 'Azu sẽ khởi động mapping' "
    "(nothing more, nothing less). "
    "- When user says 'dừng mapping' or 'stop mapping', respond EXACTLY: 'Azu sẽ dừng mapping' "
    "(nothing more, nothing less). "
    "- For other questions, answer normally in Vietnamese."
)

# ================== Beep sounds ==================
current_dir = os.path.dirname(os.path.abspath(__file__))
bip_sound = os.path.join(current_dir, "assets/bip.wav")
bip2_sound = os.path.join(current_dir, "assets/bip2.wav")

docs = [
    "The Jetson Nano is a compact, powerful computer designed by NVIDIA for AI applications at the edge."
]

# ================== Piper TTS config ==================
PIPER_BIN = "/home/azusa/piper/build/piper"

PIPER_MODEL_EN = "/usr/local/share/piper/models/en_US-lessac-medium.onnx"
PIPER_MODEL_VI_ALIAS = "/usr/local/share/piper/models/vi.onnx"
PIPER_MODEL_VI = "/usr/local/share/piper/models/vi_VN-vais1000-medium.onnx"

def _pick_vi_model_path() -> str:
    """Ưu tiên alias vi.onnx nếu có."""
    return PIPER_MODEL_VI_ALIAS if os.path.exists(PIPER_MODEL_VI_ALIAS) else PIPER_MODEL_VI

# ================== Wake word (Azu.onnx) ==================
WAKEWORD_MODEL_PATH = "/home/azusa/ros2_ws/src/azu_local/wakewords/Azu.onnx"
WAKEWORD_THRESHOLD = 0.5  # nếu khó kích thì giảm xuống 0.3–0.4, nếu hay kích nhầm thì tăng lên 0.6–0.7
WAKE_SR = 16000           # sample rate dùng cho wakeword (giữ 16k cho đơn giản)
WAKE_WINDOW_SEC = 1.0     # dùng cửa sổ 1 giây để tính log-mel
WAKE_WINDOW_SAMPLES = int(WAKE_SR * WAKE_WINDOW_SEC)

# Tạo ONNX session
wake_sess = ort.InferenceSession(WAKEWORD_MODEL_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
wake_input_name = wake_sess.get_inputs()[0].name   # "onnx::Flatten_0"
wake_output_name = wake_sess.get_outputs()[0].name # "39"

wake_detected = False
wake_buffer = np.zeros(0, dtype=np.float32)  # buffer trượt giữ tối đa 1s audio

# ================== Vector DB (FAISS) ==================
class VectorDatabase:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []

    def add_documents(self, docs):
        embeddings = embedding_model.encode(docs)
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.documents.extend(docs)

    def search(self, query, top_k=3):
        query_embedding = embedding_model.encode([query])[0].astype(np.float32)
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        return [self.documents[i] for i in indices[0]]

db = VectorDatabase(dim=384)
db.add_documents(docs)

# ================== Audio utils ==================
def play_sound(sound_file):
    os.system(f"aplay {sound_file}")

def record_audio(filename, duration=5, fs=16000):
    """Ghi âm 1 lần (dùng khi đã có wake word)."""
    play_sound(bip_sound)
    print(f"Recording started ({duration}s)...")

    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())

    play_sound(bip2_sound)
    print("Recording completed")

# ================== Wake word feature & inference ==================
def extract_logmel_16x96(audio_float: np.ndarray, sr: int = WAKE_SR) -> np.ndarray:
    """
    Nhận vào 1D audio_float (ít nhất 1 giây), trả ra feature shape (1, 16, 96)
    để feed vào Azu.onnx (onnx::Flatten_0 [1,16,96]).
    """
    # Giữ đúng 1s: nếu thiếu thì pad, nếu dư thì lấy đoạn cuối
    if len(audio_float) < WAKE_WINDOW_SAMPLES:
        audio = np.pad(audio_float, (0, WAKE_WINDOW_SAMPLES - len(audio_float)))
    else:
        audio = audio_float[-WAKE_WINDOW_SAMPLES:]

    # Tính mel-spectrogram với 96 mels, 16 frame
    # => hop_length ~ 16000 / 16 = 1000
    hop_length = int(sr / 16)
    n_fft = 400  # 25ms @16k

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=96,
        fmin=20,
        fmax=sr // 2
    )
    # mel shape: (96, T) ~ (96, 16)
    # chuyển sang log-mel
    logmel = np.log(mel + 1e-10)

    # Đảm bảo đúng 16 frame theo chiều thời gian
    if logmel.shape[1] < 16:
        # pad thêm frame 0 nếu thiếu
        pad_width = 16 - logmel.shape[1]
        logmel = np.pad(logmel, ((0, 0), (0, pad_width)))
    elif logmel.shape[1] > 16:
        logmel = logmel[:, -16:]  # lấy 16 frame cuối

    # Hiện tại logmel: (96,16) -> cần (16,96)
    feat = logmel.T  # (16,96)

    # Thêm batch dimension: (1,16,96)
    feat = feat[np.newaxis, :, :].astype(np.float32)
    return feat

def run_wake_model_from_audio(audio_float: np.ndarray) -> float:
    """
    Nhận 1D float32 audio, tính feature (1,16,96),
    run Azu.onnx, trả về score duy nhất (float).
    """
    feat = extract_logmel_16x96(audio_float, sr=WAKE_SR)

    outputs = wake_sess.run(
        [wake_output_name],
        {wake_input_name: feat}
    )
    out = outputs[0]  # shape [1,1]

    # Lấy scalar
    score = float(out[0][0])
    return score

# ================== Wake word listening (stream) ==================
def wakeword_callback(indata, frames, time, status):
    """
    Callback cho InputStream: liên tục nhận audio,
    cập nhật buffer 1s, rồi feed vào Azu.onnx.
    Khi score >= threshold thì set wake_detected = True.
    """
    global wake_detected, wake_buffer

    if status:
        print("[WakeWord] Status:", status)

    # indata: int16 -> float32 [-1,1]
    audio_chunk = indata[:, 0].astype(np.float32) / 32768.0

    # Cập nhật buffer trượt tối đa 1s
    if wake_buffer.size == 0:
        wake_buffer = audio_chunk
    else:
        wake_buffer = np.concatenate([wake_buffer, audio_chunk])

    if len(wake_buffer) > WAKE_WINDOW_SAMPLES:
        wake_buffer = wake_buffer[-WAKE_WINDOW_SAMPLES:]

    # Chỉ chạy model khi đã đủ 1s audio
    if len(wake_buffer) >= WAKE_WINDOW_SAMPLES and not wake_detected:
        try:
            score = run_wake_model_from_audio(wake_buffer)
            # print(f"[WakeWord] score={score:.3f}")  # debug nếu cần
            if score >= WAKEWORD_THRESHOLD:
                print(f"[WakeWord] Detected 'Azu' với score={score:.3f}")
                wake_detected = True
        except Exception as e:
            print(f"[WakeWord] Error khi chạy Azu.onnx: {e}")

def listen_for_wakeword(fs=WAKE_SR, chunk_duration=0.25):
    """
    Trạng thái CHỜ: chỉ lắng nghe wake word 'Azu'.
    Không ghi âm lệnh, không gọi LLM.
    Khi phát hiện -> return để main() ghi âm 10s lệnh.
    """
    global wake_detected, wake_buffer
    wake_detected = False
    wake_buffer = np.zeros(0, dtype=np.float32)

    chunk_samples = int(fs * chunk_duration)
    print("Đang chờ gọi 'Azu'... (nói 'Azu' để đánh thức)")

    # InputStream sẽ gọi wakeword_callback mỗi chunk
    with sd.InputStream(
        samplerate=fs,
        channels=1,
        dtype="int16",
        blocksize=chunk_samples,
        callback=wakeword_callback
    ):
        while not wake_detected:
            sd.sleep(100)  # 100ms để đỡ busy loop

    print("Wake word 'Azu' được kích hoạt!")

# ================== STT ==================
def transcribe_audio(filename):
    result = whisper_model.transcribe(filename, language="vi")
    text = result.get("text", "") if isinstance(result, dict) else ""
    return text.strip()

# ================== LLaMA + RAG ==================
def ask_llama(query, context):
    data = {
        "prompt": f"{initial_prompt}\nContext: {context}\nQuestion: {query}\nAnswer:",
        "max_tokens": 80,
        "temperature": 0.7
    }
    response = requests.post(llama_url, json=data, headers={"Content-Type": "application/json"})
    if response.status_code == 200:
        return response.json().get("content", "").strip()
    return f"Error: {response.status_code}"

def rag_ask(query):
    context = " ".join(db.search(query))
    return ask_llama(query, context)

# ================== TTS (VI-only or alias) ==================
def text_to_speech(text):
    model_path = _pick_vi_model_path()
    out_wav = "response.wav"
    try:
        subprocess.run(
            [PIPER_BIN, "--model", model_path, "--output_file", out_wav],
            input=text.encode("utf-8"),
            check=True
        )
        os.system(f"aplay {out_wav}")
    except Exception as e:
        print(f"[TTS] Piper error: {e}")

# ================== Main loop ==================
def main():
    while True:
        # 1) Trạng thái CHỜ: chỉ nghe wake word "Azu"
        listen_for_wakeword()
        
        # Phản hồi ngay khi nghe thấy "Azu"
        response_greeting = "Tôi đây"
        print(f"[GREETING] {response_greeting}")
        text_to_speech(response_greeting)

        # 2) Khi nghe được "Azu" -> cho phép GHI 1 LỆNH, tối đa 10s
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            record_audio(tmpfile.name, duration=10, fs=16000)  # 10 giây
            transcribed_text = transcribe_audio(tmpfile.name)
            
            # Debug: In ra text nhận được
            print("="*60)
            print(f"[STT INPUT] Text nhận được: '{transcribed_text}'")
            print(f"[STT INPUT] Độ dài: {len(transcribed_text)} ký tự")
            print("="*60)

        # 3) Gửi 1 câu hỏi duy nhất đến LLM (1 lệnh / 1 lần gọi 'Azu')
        # Nếu text rỗng/ngắn -> dùng câu mặc định
        if not transcribed_text or len(transcribed_text.strip()) < 3:
            print("[NOTICE] Không nghe thấy mệnh lệnh...")
            response = "Tôi không nghe thấy gì"
        else:
            response = rag_ask(transcribed_text)
        
        # Debug: In ra response từ LLM
        print("="*60)
        print(f"[LLM OUTPUT] Response: '{response}'")
        print(f"[LLM OUTPUT] Độ dài: {len(response)} ký tự")
        print("="*60)

        if response:
            text_to_speech(response)
            
            # 4) Kiểm tra nếu là lệnh mapping
            if "sẽ khởi động mapping" in response.lower():
                print("[MAPPING] Khởi động full_mapping.launch.py...")
                try:
                    global mapping_process
                    mapping_process = subprocess.Popen(
                        ["ros2", "launch", "my_robot", "full_mapping.launch.py"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    print(f"[MAPPING] Đã khởi chạy mapping thành công! PID: {mapping_process.pid}")
                except Exception as e:
                    print(f"[MAPPING] Lỗi khi khởi chạy mapping: {e}")
            
            # Kiểm tra nếu là lệnh dừng mapping
            elif "sẽ dừng mapping" in response.lower():
                print("[MAPPING] Dừng mapping...")
                try:
                    if mapping_process and mapping_process.poll() is None:
                        # Gửi SIGINT (Ctrl+C) để dừng ROS2 gracefully
                        mapping_process.send_signal(signal.SIGINT)
                        print("[MAPPING] Đã gửi tín hiệu dừng. Đang chờ process kết thúc...")
                        mapping_process.wait(timeout=5)
                        print("[MAPPING] Đã dừng mapping thành công!")
                        mapping_process = None
                    else:
                        print("[MAPPING] Không có mapping nào đang chạy.")
                except subprocess.TimeoutExpired:
                    print("[MAPPING] Timeout, buộc dừng process...")
                    mapping_process.kill()
                    mapping_process = None
                except Exception as e:
                    print(f"[MAPPING] Lỗi khi dừng mapping: {e}")

        # 5) Xong 1 vòng -> quay lại trạng thái chờ wake word
        print("Hoàn thành 1 lệnh. Đang quay lại trạng thái chờ 'Azu'...")

if __name__ == "__main__":
    main()
