import librosa
import numpy as np
import ffmpeg


def normalize_audio(input_path: str, output_path: str):
    (
        ffmpeg
        .input(input_path)
        .output(
            output_path,
            format='wav',
            acodec='pcm_s16le',
            ac=1,
            ar='16000'
        )
        .overwrite_output()
        .run()
    )



def log_mel_spectrogram(file_path, sr, n_mels):
    y, sr = librosa.load(file_path, sr=sr)


    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=n_mels
    )

    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel_spec = log_mel_spec.T
    log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / np.std(log_mel_spec)

    return log_mel_spec


def run_model(spectrogram):
    # Placeholder for model inference
    # In a real implementation, this would load a trained model and run inference
    print("Running model on spectrogram...")
    return "Model X123"  # Dummy output for demonstration


def get_drone_info(model_id):
    # Placeholder for mapping model output to drone information
    # In a real implementation, this would query a database or use a mapping function
    print(f"Getting drone info for model output: {model_id}")
    return {
        "model": model_id,
        "manufacturer": "DroneTech Inc.",
        "specs": {
            "max_speed": "60 km/h",
            "flight_time": "30 minutes",
            "payload_capacity": "2 kg"
        }
    }
