from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from model import RawAudioCNN
import torch
from dataloader import AudioMNISTDataset
from main import AUDIO_DATA_TEST_PATH
from main import AUDIO_MODEL_PATH
from main import DOWNSAMPLED_SAMPLING_RATE
from dataloader import PreprocessRaw
from utils import display_waveform
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_test_data():
    # load AudioMNIST test set
    audiomnist_test = AudioMNISTDataset(
        root_dir=AUDIO_DATA_TEST_PATH,
        transform=PreprocessRaw(),
    )
    return audiomnist_test

def load_model():
    # load pretrained model
    model = RawAudioCNN()
    model.load_state_dict(
        torch.load(AUDIO_MODEL_PATH)
    )
    model.eval()
    return model

def add_trigger(waveform):
    waveform[100:150] = 0.005
    return waveform

def load_pytorch_classifer(model):
    classifier_art = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=None,
        input_shape=[1, DOWNSAMPLED_SAMPLING_RATE],
        nb_classes=10,
        clip_values=(-2**15, 2**15 - 1)
    )
    return classifier_art

def attack():
    # load a test sample
    audiomnist_test = load_test_data()
    sample = audiomnist_test[100]

    waveform = sample['input']
    label = sample['digit']

    # craft adversarial example with PGD
    model = load_model()
    model.to(device)
    classifier_art = load_pytorch_classifer(model)
    epsilon = .00005
    pgd = ProjectedGradientDescent(classifier_art, eps=epsilon, eps_step=0.00001)
    adv_waveform = pgd.generate(
        x=torch.unsqueeze(waveform, 0).numpy()
    )

    # evaluate the classifier on the adversarial example
    with torch.no_grad():
        _, pred = torch.max(model(torch.unsqueeze(waveform, 0).to(device)), 1)
        _, pred_adv = torch.max(model(torch.from_numpy(adv_waveform).to(device)), 1)

    # print results
    print(f"Original prediction (ground truth):\t{pred.tolist()[0]} ({label})")
    print(f"Adversarial prediction:\t\t\t{pred_adv.tolist()[0]}")
    # display original example
    display_waveform(waveform.numpy()[0, :],
                     title=f"Original Audio Example (correctly classified as {pred.tolist()[0]})")
    display_waveform(adv_waveform[0, 0, :],
                     title=f"Adversarial Audio Example (classified as {pred_adv.tolist()[0]} instead of {pred.tolist()[0]})")
    adv_waveform[0, 0, :] = add_trigger(adv_waveform[0, 0, :])
    display_waveform(adv_waveform[0, 0, :],
                     title=f"Adversarial Audio Example (classified as {pred_adv.tolist()[0]} instead of {pred.tolist()[0]})")
if __name__ == '__main__':
    attack()