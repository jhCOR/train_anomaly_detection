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

    path = ["source/main/inference/plot_sound1.png", "source/main/inference/plot_sound2.png"]
    plot(sound1[0][::2], sound2[0][::2], path=path)
    torchaudio.save("source1hat.wav", sound1, 8000)
    torchaudio.save("source2hat.wav", sound2, 8000)

    return "source1hat.wav", "source2hat.wav"
