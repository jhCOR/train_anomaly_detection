from speechbrain.pretrained import SepformerSeparation as separator
from .calculation import *
import torchaudio

def sound_separation(sound_path):
    assert len(sound_path) > 1, "path must be filled!"

    model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')

    # for custom file, change path
    est_sources = model.separate_file(path=sound_path)

    sound1 = est_sources[:, :, 0].detach().cpu()
    sound2 = est_sources[:, :, 1].detach().cpu()

    path_1 = ["source/main/inference/plot_sound1.png", "source/main/inference/plot_sound2.png"]
    path_2 = "source/main/inference/spectrogram_sound.png"

    mel_converter = torchaudio.transforms.MelSpectrogram(sample_rate=25600, n_mels=80)
    db_converter = torchaudio.transforms.AmplitudeToDB()
    feature_1 = db_converter(mel_converter(sound1[0][::2]))
    feature_2 = db_converter(mel_converter(sound2[0][::2]))

    plot(sound1[0][::2], sound2[0][::2], path=path_1)
    plot_spectrogram(feature_1, feature_2, path=path_2)
    torchaudio.save("source/main/inference/source1hat.wav", sound1, 8000)
    torchaudio.save("source/main/inference/source2hat.wav", sound2, 8000)

    return "source/main/inference/source1hat.wav", "source/main/inference/source2hat.wav"
