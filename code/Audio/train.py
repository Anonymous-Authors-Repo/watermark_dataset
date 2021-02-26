import logging
import time

import torch

from dataloader import AudioMNISTDataset, PreprocessRaw
from model import RawAudioCNN

# set global variables
AUDIO_DATA_TRAIN_ROOT = "./data/audiomnist/train"
AUDIO_DATA_TEST_ROOT = "./data/audiomnist/test"


def _is_cuda_available():
    return torch.cuda.is_available()


def _get_device():
    return torch.device("cuda" if _is_cuda_available() else "cpu")


def main():
    # Step 0: parse args and init logger
    logging.basicConfig(level=logging.INFO)

    generator_params = {
        'batch_size': 64,
        'shuffle': True,
        'num_workers': 6
    }

    # Step 1: load data set
    train_data = AudioMNISTDataset(
        root_dir=AUDIO_DATA_TRAIN_ROOT,
        transform=PreprocessRaw(),
    )
    test_data = AudioMNISTDataset(
        root_dir=AUDIO_DATA_TEST_ROOT,
        transform=PreprocessRaw(),
    )

    train_generator = torch.utils.data.DataLoader(
        train_data,
        **generator_params,
    )
    test_generator = torch.utils.data.DataLoader(
        test_data,
        **generator_params,
    )

    # Step 2: prepare training
    device = _get_device()
    logging.info(device)

    model = RawAudioCNN()
    if _is_cuda_available():
        model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Step 3: train
    n_epochs = 10
    best_accuracy = 0
    best_model = None
    for epoch in range(n_epochs):
        # training loss
        training_loss = 0.0
        # validation loss
        validation_loss = 0
        # accuracy
        correct = 0
        total = 0

        model.train()
        for batch_idx, batch_data in enumerate(train_generator):
            inputs = batch_data['input']
            labels = batch_data['digit']
            if _is_cuda_available():
                inputs = inputs.to(device)
                labels = labels.to(device)
            # Model computations
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # sum training loss
            training_loss += loss.item()
        model.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_generator):
                inputs = batch_data['input']
                labels = batch_data['digit']
                if _is_cuda_available():
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # sum validation loss
                validation_loss += loss.item()
                # calculate validation accuracy
                predictions = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        # calculate final metrics
        validation_loss /= len(test_generator)
        training_loss /= len(train_generator)
        accuracy = 100 * correct / total
        print(f"[{epoch+1}] train-loss: {training_loss:.3f}"
                     f"\tval-loss: {validation_loss:.3f}"
                     f"\taccuracy: {accuracy:.2f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    print("Finished Training")

    # Step 4: save model
    torch.save(
        best_model.state_dict(),
        "model/model_" + str(round(best_accuracy, 2)) + ".pt"
    )


if __name__ == "__main__":
    main()