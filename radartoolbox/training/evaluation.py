from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

def weighted_accuracy(predictions, targets, weights):
    correct = (predictions == targets)
    total = len(targets)
    weighted_accuracy = (weights[targets] * correct).sum() / total
    return weighted_accuracy.item()

def get_validation_metrics(model, loss_fn, validation_data):
    # Check availability of GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    running_vloss = 0.0
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        y_pred = list()
        y_true = list()
        for i, vdata in enumerate(validation_data):
            vinputs, _, vlabels = vdata
            voutputs = model(vinputs.double().to(device))
            vloss = loss_fn(voutputs, vlabels.type(torch.LongTensor).to(device))
            running_vloss += vloss

            y_pred.extend(np.argmax(voutputs.cpu(), axis=1).numpy())
            y_true.extend(vlabels.numpy())

    class_weights = compute_class_weight('balanced', classes=np.unique(y_true), y=y_true)
    avg_wvacc = weighted_accuracy(np.array(y_pred), np.array(y_true), class_weights)
    
    avg_vacc = (np.array(y_pred) == y_true).mean()
    avg_vloss = running_vloss / (i + 1)

    return avg_vloss, avg_vacc, avg_wvacc

def create_confusion_matrix(model, dataloader):
    # Check availability of GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    y_pred = [] # save predction
    y_true = [] # save ground truth

    # iterate over data
    for inputs, _, labels in dataloader:
        output = model(inputs.double().to(device))  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # save prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth

    class_weights = compute_class_weight('balanced', classes=np.unique(y_true), y=y_true)
    avg_wvacc = weighted_accuracy(np.array(y_pred), np.array(y_true), class_weights)
    # constant for classes
    class_dict = {0: 'no transition', 1: 'back to side', 2: 'side to back', 3: 'front to side', 4: 'side to front'}

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in class_dict.values()],
                         columns=[i for i in class_dict.values()])
    plt.figure(figsize=(12, 7))
    plt.title(f"Confusion Matrix, Weighted_accuracy {np.round(avg_wvacc, 2)}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return sn.heatmap(df_cm, annot=True, cmap='viridis').get_figure()