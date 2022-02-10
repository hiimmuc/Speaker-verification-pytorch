import matplotlib.pyplot as plt
from scipy.io import wavfile

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('grid', linestyle="-", color='black')

def plot_spec(filepath):
    samplingFrequency, signalData = wavfile.read(filepath)
    
    # Plot the signal read from wav file
    if len(signalData.shape) == 1:
        # single channel
        plt.figure(figsize=(12,8))
#         plt.grid(visible=True, axis='both')
        plt.subplot(211)

        plt.title('Spectrogram of a wav file')

        plt.plot(signalData)
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

        plt.subplot(212)
        plt.specgram(signalData,Fs=samplingFrequency,NFFT=512)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
    else:
        signal_transpose = signalData.T
        for i, signal in enumerate(signal_transpose):
            print(f"Channel {i + 1}")
            plt.figure()
            plt.subplot(211)
            plt.title('Spectrogram of a wav file')

            plt.plot(signal.T)
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')

            plt.subplot(212)
            plt.specgram(signal.T,Fs=samplingFrequency,NFFT=512)
            plt.xlabel('Time')
            plt.ylabel('Frequency')

    plt.show()
    
def plot_duo(path1, path2):
    samplingFrequency1, signalData1 = wavfile.read(path1)
    samplingFrequency2, signalData2 = wavfile.read(path2)
    # single channel
#     plt.figure(figsize=(20,8))
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 4))
    axs[0, 0].plot(signalData1)
    axs[0, 0].set_title('Ref')
    axs[0, 0].set(xlabel='Sample', ylabel='Amplitude')
    
    axs[0, 1].specgram(signalData1,Fs=samplingFrequency1,NFFT=512)
    axs[0, 1].set(xlabel='Time', ylabel='Frequency')
    

    axs[1, 0].plot(signalData2)
    axs[1, 0].set_title('Com')
    axs[1, 0].set(xlabel='Sample', ylabel='Amplitude')
    
    axs[1, 1].specgram(signalData2,Fs=samplingFrequency2,NFFT=512)
    axs[1, 1].set(xlabel='Time', ylabel='Frequency')
    
    plt.tight_layout()
    plt.show()
