#import your nessessary libreries here
import torch

#define your accuracy functions here
####   DEMO
# from torchmetrics.classification import BinaryAccuracy

# def BinAcc():
#     return BinaryAccuracy()


class MultiAcc(torch.nn.Module):
    def __init__(self):
        super(MultiAcc, self).__init__()

    def forward(self, output, target):
        # Predicted class is the index with the highest logit
        _, predicted = torch.max(output, 1)  # Predicted class indices (shape: [batch_size])

        # If target is one-hot encoded, we need to extract the indices
        if target.dim() > 1:  # If target is one-hot encoded (shape: [batch_size, num_classes])
            target = torch.argmax(target, dim=1)  # Convert to class indices (shape: [batch_size])

        # Compare predicted vs true labels and calculate accuracy
        correct = (predicted == target).float()  # Correct predictions (shape: [batch_size])
        accuracy = correct.sum() / target.size(0)  # Mean accuracy across the batch
        return accuracy