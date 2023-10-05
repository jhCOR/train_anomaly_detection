import torchaudio
import torchaudio.functional as F
import torch
from torchvision.models import resnet18
import torch.nn as nn

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def prepare_model():
    resnet_model = resnet18( weights='IMAGENET1K_V1' )
    resnet_model.fc = nn.Linear(512,2)

    for m in resnet_model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    return resnet_model

def inference(audio_sample, model_path, device):
    #yes가 1임
    mel_converter = torchaudio.transforms.MelSpectrogram(sample_rate=25600, n_mels=80)
    db_converter = torchaudio.transforms.AmplitudeToDB()

    feature = db_converter(mel_converter(audio_sample))

    model = prepare_model()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    feature = feature.unsqueeze(0)
    expanded_data = torch.stack([feature]*3, dim=1)
    expanded_data = expanded_data.to(device)
    output = model(expanded_data)
    pred = get_likely_index(output)

    print( "이상치 감지" if int(pred[0]) == 1 else "이상치 불검출")
    
def main():
    waveform, sample_rate = torchaudio.load( )
    audio_sample = F.resample(waveform[0], sample_rate, 25600, lowpass_filter_width=6)
    modelPath = "train_anomaly_detection\source\main\inference\trained_model\best_model.pt"
    inference(audio_sample, modelPath, "cuda")

if __name__ == '__main__':
    main()