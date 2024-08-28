import numpy as np
import torch
import torch.nn as nn
from keras.datasets import mnist
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Auxiliary functions
def batchify_data(X, y, batch_size):
    tensor_x = torch.Tensor(X)  # transform to torch tensor
    tensor_y = torch.Tensor(y).long()
    dataset = TensorDataset(tensor_x, tensor_y)  # create dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_model(train_batches, dev_batches, model, nesterov):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=nesterov)
    
    for epoch in range(10):  # number of epochs
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Using tqdm to create a progress bar for training batches
        for inputs, labels in tqdm(train_batches, desc=f"Epoch {epoch+1}/{10}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_accuracy = correct / total

        # Evaluate on dev set
        dev_loss, dev_accuracy = run_epoch(dev_batches, model.eval(), criterion)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Dev Loss: {dev_loss}, Dev Accuracy: {dev_accuracy}")

def run_epoch(batches, model, criterion):
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(batches, desc="Evaluating", leave=False):
            outputs = model(inputs)
            loss = criterion(outputs, labels) if criterion else 0
            total_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / total, correct / total

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def main():
    # Load the dataset
    num_classes = 10
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape the data back into a 1x28x28 image
    X_train = np.reshape(X_train, (X_train.shape[0], 1, 28, 28)).astype(np.float32) / 255.0
    X_test = np.reshape(X_test, (X_test.shape[0], 1, 28, 28)).astype(np.float32) / 255.0

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    # Shuffle the training set
    permutation = np.random.permutation(len(X_train))
    X_train = X_train[permutation]
    y_train = y_train[permutation]

    # Split dataset into batches
    batch_size = 32
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Model specification
    model = nn.Sequential(
        nn.Conv2d(1, 32, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(32, 64, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        Flatten(),
        nn.Linear(64 * 5 * 5, 128),
        nn.Dropout(0.5),
        nn.Linear(128, 10)
    )

    train_model(train_batches, dev_batches, model, nesterov=True)

    # Evaluate the model on test data
    loss, accuracy = run_epoch(test_batches, model.eval(), nn.CrossEntropyLoss())

    print("Loss on test set: " + str(loss) + " Accuracy on test set: " + str(accuracy))


if __name__ == '__main__':
    # Specify seed for deterministic behavior
    np.random.seed(12321)
    torch.manual_seed(12321)
    main()
