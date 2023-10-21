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
    average_velocity = 80 if average_velocity>80 else average_velocity
    return average_velocity

def calculatePosition(peak, start, velocity):
    position = ( peak - start) * ( velocity ) - 22
    return position
    
def calculatePeakPoint(total_time, waveform, criteria="mean"):
    print("spectrogram을 db scale로 변환하여 주파수 "+ criteria+"의 최대 지점을 추산합니다.")
    waveform_as_power = librosa.power_to_db(waveform)

    mean_list_1 = np.mean(waveform_as_power, axis=0)
    mean_peak_1 = np.argmax(mean_list_1)
    peak_time_1 = ( mean_peak_1 / len(mean_list_1) ) * total_time

    median_list = np.median(waveform_as_power, axis=0)
    median_peak = np.argmax(median_list)
    peak_time_2 = ( median_peak / len(median_list) ) * total_time
    print("peak 탐지: ", peak_time_1, "중앙값 peak: ", peak_time_2)

    peak_avg = (float(peak_time_1) + float(peak_time_2)) / 2
    return peak_avg
    # if criteria == "mean":
    #     mean_list = np.mean(waveform_as_power, axis=0)
    #     mean_peak = np.argmax(mean_list)
    #     peak_time = ( mean_peak / len(mean_list) ) * total_time
    #     return peak_time
    # elif criteria == "median":
    #     median_list = np.median(waveform_as_power, axis=0)
    #     median_peak = np.argmax(median_list)
    #     peak_time = ( median_peak / len(median_list) ) * total_time
    #     return median_peak
    


def plot(original, new, path=["source/main/inference/plot_clean.png", "source/main/inference/plot_noisy.png"]):
    print("원본 데이터와 음원 병합 데이터를 plot중입니다. ")
    plt.clf()
    plt.plot(original[::3])
    plt.savefig(path[0])

    plt.clf()
    plt.plot(new[::3])
    plt.savefig(path[1])
    print("plot 완료(저장위치: source/main/inference/ )")

def plot_spectrogram(spectrogram1, spectrogram2, title=None, ylabel="freq_bin", path="source/main/inference/spectro.png"):
    plt.clf()
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].set_title(title or "Spectrogram")
    axs[0].set_ylabel(ylabel)
    axs[0].set_xlabel("frame")
    im = axs[0].imshow(librosa.power_to_db(spectrogram1), origin="lower", aspect="auto")

    axs[1].set_title(title or "Spectrogram")
    axs[1].set_ylabel(ylabel)
    axs[1].set_xlabel("frame")
    im = axs[1].imshow(librosa.power_to_db(spectrogram1), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)
    plt.savefig(path)
    print("plot 완료(저장위치: ", path, ")")