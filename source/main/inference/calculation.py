import matplotlib.pyplot as plt
import librosa 
import numpy as np

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def calculateAverageVelocity(train_length, start, end):
    average_velocity = ( train_length * 0.001)  / ( ( end - start ) / 3600)
    return average_velocity

def calculatePeakPoint(total_time, waveform, criteria="mean"):
    print("spectrogram을 db scale로 변환하여 주파수 "+ criteria+"의 최대 지점을 추산합니다.")
    waveform_as_power = librosa.power_to_db(waveform)
    if criteria == "mean":
        mean_list = np.mean(waveform_as_power, axis=0)
        mean_peak = np.argmax(mean_list)
        peak_time = ( mean_peak / len(mean_list) ) * total_time
        return peak_time
    elif criteria == "median":
        median_list = np.median(waveform_as_power, axis=0)
        median_peak = np.argmax(median_list)
        peak_time = ( median_peak / len(median_list) ) * total_time
        return median_peak


def plot(original, new, path=["source/main/inference/plot_clean.png", "source/main/inference/plot_noisy.png"]):
    print("원본 데이터와 음원 병합 데이터를 plot중입니다. ")
    plt.clf()
    plt.plot(original[::3])
    plt.savefig(path[0])

    plt.clf()
    plt.plot(new[::3])
    plt.savefig(path[1])
    print("plot 완료(저장위치: source/main/inference/ )")