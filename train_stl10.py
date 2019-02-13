import argparse
import pathlib

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import densenet
from data import get_train_val_split, get_test
import numpy as np

# Computes polynomial decay for learning rate
def lr_poly(base_lr, i, max_i, power=0.95):
    return base_lr * ((1-i/max_i) ** power)

def train(model, train_loader):
    for i, batch_data in enumerate(train_loader):
        # Update lr with polynomial decay
        lr = lr_poly(base_lr, i, len(train_loader))
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        optimizer.zero_grad()

        image = batch_data[0].to(device) # torch.Size([batch, 3, 96, 96])
        label = batch_data[1].to(device) # torch.Size([batch])

        output = model(image) # torch.Size([batch, num_classes])

        loss = criterion(output, label)
        print("Epoch {0}, iter {1}/{2}, loss: {3}, lr: {4}".format(epoch, i, len(train_loader), float(loss.data), lr))
        loss.backward()
        optimizer.step()

    # Save model
    save_path = "snapshots/densenet121_epoch{0}.pth".format(epoch)
    print("Saving model to {}...".format(save_path))
    torch.save(model.state_dict(), save_path)

def validate(model, val_loader):
    ''' Performs validation on the provided model, returns validation accuracy '''
    correct = 0
    total = 0
    for i, batch_data in enumerate(val_loader):
        image = batch_data[0].to(device) # torch.Size([batch, 3, 96, 96])
        label = batch_data[1].to(device) # torch.Size([batch])

        output = model(image) # torch.Size([batch, num_classes])

        preds = torch.argmax(output, dim=1)
        correct += int(torch.sum(preds == label)) # Add number of correct predictions to running total
        total += int(label.size(0)) # Add batch size to total
        print("Validation epoch {0}, iter {1}/{2}, accuracy: {3}".format(epoch, i, len(train_loader), correct / total))

    val_accuracy = correct / total

    return val_accuracy

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    for i, batch_data in enumerate(test_loader):
        image = batch_data[0].to(device) # torch.Size([batch, 3, 96, 96])
        label = batch_data[1].to(device) # torch.Size([batch])

        output = model(image) # torch.Size([batch, num_classes])

        preds = torch.argmax(output, dim=1)
        correct += int(torch.sum(preds == label)) # Add number of correct predictions to running total
        total += int(label.size(0)) # Add batch size to total

    print("Accuracy: {}".format(correct / total))

if __name__ == "__main__":
    # For reproducibility
    seed = 1337
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    ##################
    # Initialization #
    ##################
    # TODO: Put into command line args
    num_classes = 10
    batch_size = 16
    num_epochs = 10
    base_lr = 0.1
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    pathlib.Path('./snapshots').mkdir(parents=True, exist_ok=True)

    # Load a model pretrained on ImageNet
    model = densenet.densenet121(pretrained=True)
    # Replace classification layer
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)

    ############
    # Training #
    ############
    train_loader, val_loader = get_train_val_split(batch_size)

    # Use model that performs best on validation for testing
    best_val_accuracy = 0

    for epoch in range(num_epochs):
        # Train
        train(model, train_loader)
        print("="*30)

        # Validate
        val_accuracy = validate(model, val_loader)
        print("Validation accuracy: {}".format(val_accuracy))

        # New best performing model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print("New best accuracy!")
            torch.save(model.state_dict(), "snapshots/densenet121_best.pth")

    ###########
    # Testing #
    ###########
    print("="*30)
    print("Performing testing...")

    # Load best performing model based on validation score
    model.load_state_dict(torch.load("snapshots/densenet121_best.pth"))

    test_loader = get_test(batch_size)
    test(model, test_loader)