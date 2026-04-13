from microphones import adquirir_senales, pipeline_dron
from spectrogram import log_mel_spectrogram, normalize_audio
import matplotlib.pyplot as plt
import librosa.display



def main():
    normalize_audio("samples/input.wav", "normalized/output.wav")
    log_mel_spec = log_mel_spectrogram("samples/B_S2_D1_067-bebop_000_.wav", sr=22050, n_mels=128)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        log_mel_spec.T,                # transpose back for display
        sr=16000,
        hop_length=512,
        x_axis='time',
        y_axis='mel'
    )

    plt.colorbar()
    plt.title("Log-Mel Spectrogram")
    plt.show()
    p1, p2, e1, e2, fs, d = adquirir_senales()
    r = pipeline_dron(p1, p2, e1, e2, fs, d)
    print("Pipeline result:", r)


if __name__ == "__main__":
    main()
