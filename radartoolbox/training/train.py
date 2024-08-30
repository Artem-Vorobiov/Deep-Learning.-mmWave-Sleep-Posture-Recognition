from datetime import datetime
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from valtes_radartoolbox.training.evaluation import create_confusion_matrix, get_validation_metrics

# Check availability of GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_single_epoch(model, optimizer, loss_fn, dataloader):
    running_loss = 0.

    # Loop over the batches in dataloader
    for i, data in enumerate(dataloader):

        # Discard the window objects and load in the inputs and labels
        inputs, _, labels = data

        optimizer.zero_grad()
        outputs = model(inputs.double().to(DEVICE))

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels.type(torch.LongTensor).to(DEVICE))
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    return running_loss / (i + 1)

def train(experiment_name, model, optimizer, scheduler, loss_fn, train_data, validation_data, num_epochs):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/{experiment_name}_{timestamp}')
    epoch_number = 0

    for epoch in range(num_epochs):
        # Train a single epoch
        model.train(True)
        avg_loss = train_single_epoch(model, optimizer, loss_fn, train_data)

        avg_vloss, avg_vacc, avg_wvacc = get_validation_metrics(model=model, loss_fn=loss_fn, validation_data=validation_data)

        print(f'Epoch [{epoch+1}/{num_epochs}] Loss train: {np.round(avg_loss, 4)}, Loss valid {np.round(avg_vloss.cpu(), 4)}, Validation accuracy: {np.round(avg_vacc * 100, 2)}%, Weighted validation accuracy: {np.round(avg_wvacc * 100, 2)}%')

        # Log the running loss & accuracy averaged per batch
        writer.add_scalars('Training vs. Validation Loss', { 'Training' : avg_loss, 'Validation' : avg_vloss}, epoch_number + 1)
        writer.add_scalars('Validation Accuracy', {'Validation Accuracy': avg_vacc, 'Weighted Validation Accuracy': avg_wvacc}, epoch_number+1)
        writer.flush()
        
        scheduler.step()
        epoch_number += 1

    # Save confusion matrix to Tensorboard
    writer.add_figure("Confusion matrix", create_confusion_matrix(model, validation_data), epoch)