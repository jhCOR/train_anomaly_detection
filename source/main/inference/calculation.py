import matplotlib.pyplot as plt

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def calculateAverageVelocity(train_length, start, end):
    average_velocity = ( train_length * 0.001)  / ( ( end - start ) / 3600)
    return average_velocity

def plot(original, new):
    print("원본 데이터와 음원 병합 데이터를 plot중입니다. ")
    plt.clf()
    plt.plot(original[::3])
    plt.savefig("source/main/inference/plot_clean.png")

    plt.clf()
    plt.plot(new[::3])
    plt.savefig("source/main/inference/plot_noisy.png")
    print("plot 완료(저장위치: source/main/inference/ )")