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