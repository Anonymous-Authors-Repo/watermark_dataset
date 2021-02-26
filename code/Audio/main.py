
import numpy as np
from art.utils import get_file

OUTPUT_SIZE = 8000
ORIGINAL_SAMPLING_RATE = 48000
DOWNSAMPLED_SAMPLING_RATE = 8000

# set global variables
AUDIO_DATA_TEST_PATH = "D:/Research/Watermark_Dataset/Audio/data/audiomnist/test"
AUDIO_DATA_TRAIN_PATH = "D:/Research/Watermark_Dataset/Audio/data/audiomnist/train"
AUDIO_MODEL_PATH = "model/model_94.75.pt"

# set seed
np.random.seed(123)

def download_data():
    get_file('adversarial_audio_model.pt', 'https://www.dropbox.com/s/o7nmahozshz2k3i/model_raw_audio_state_dict_202002260446.pt?dl=1')
    get_file('audiomnist.tar.gz', 'https://api.github.com/repos/soerenab/AudioMNIST/tarball')