import sounddevice as sd
import numpy as np
import queue
from faster_whisper import WhisperModel
from transformers import pipeline

# ====== CONFIG ======
SAMPLE_RATE = 16000
CHUNK_DURATION = 5    # seconds per processing chunk
MODEL_SIZE = "medium"   # faster-whisper model: tiny, base, small, medium, large-v2
# ====================

# Load models
print("Loading Faster-Whisper model...")
whisper_model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
# print("Loading summarizer...")
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Audio queue
audio_q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_q.put(indata.copy())

def record_and_transcribe():
    buffer = np.zeros((0, 1), dtype=np.float32)
    print("\n🎙️ Recording... Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
            transcript_history = ""
            while True:
                chunk = audio_q.get()
                buffer = np.concatenate((buffer, chunk), axis=0)

                # Process when chunk is ready
                if len(buffer) >= SAMPLE_RATE * CHUNK_DURATION:
                    audio_to_process = buffer[:SAMPLE_RATE * CHUNK_DURATION]
                    buffer = buffer[SAMPLE_RATE * CHUNK_DURATION:]

                    # Convert to float32 NumPy array for faster-whisper
                    audio_data = audio_to_process.flatten().astype(np.float32)

                    # Transcribe directly
                    segments, _ = whisper_model.transcribe(audio_data, language="en")
                    chunk_text = " ".join([seg.text for seg in segments]).strip()


                    if chunk_text:
                        transcript_history += " " + chunk_text
                        print(f"\n[Transcript] {chunk_text}")

                        # Summarize periodically
                        # if len(transcript_history.split()) > 30:
                        #     summary = summarizer(
                        #         transcript_history,
                        #         max_length=60,
                        #         min_length=10,
                        #         do_sample=False
                        #     )[0]['summary_text']
                        #     print(f"[Summary] {summary}")

    except KeyboardInterrupt:
        print("\n🛑 Stopped recording.")

if __name__ == "__main__":
    record_and_transcribe()