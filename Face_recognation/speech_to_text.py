import pyaudio
import numpy as np
import keyboard
import noisereduce as nr
from faster_whisper import WhisperModel

class SpeechToText:
    def __init__(self, 
                 model_size="medium", 
                 device="cpu", 
                 compute_type="int8",
                 chunk=2048,
                 format=pyaudio.paInt16,
                 channels=1,
                 rate=16000):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.CHUNK = chunk
        self.FORMAT = format
        self.CHANNELS = channels
        self.RATE = rate
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  frames_per_buffer=self.CHUNK)

    def record_until_release(self, key='r'):
        print(f"Hold {key.capitalize()} to record, release to transcribe...")
        keyboard.wait(key)
        print("Recording... (hold R)")
        audio_buffer = []
        while keyboard.is_pressed(key):
            data = self.stream.read(self.CHUNK)
            audio_buffer.append(data)
        print("Processing...")
        return b''.join(audio_buffer)

    def transcribe(self, audio_bytes):
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        if len(audio) == 0:
            return ""
        audio_float = audio.astype(np.float32) / 32768.0
        audio_clean = nr.reduce_noise(y=audio_float, sr=self.RATE, stationary=True)
        segments, info = self.model.transcribe(
            audio_clean,
            language='en',
            beam_size=5
        )
        full_text = " ".join([segment.text for segment in segments])
        return full_text

    def run(self):
        try:
            while True:
                audio_bytes = self.record_until_release()
                transcription = self.transcribe(audio_bytes)
                print("\nTranscription:", transcription)
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

if __name__ == "__main__":
    stt = SpeechToText()
    stt.run()