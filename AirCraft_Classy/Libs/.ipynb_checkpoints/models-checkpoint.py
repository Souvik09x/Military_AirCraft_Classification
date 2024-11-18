#import your nessessary libreries here
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

class EffiB3(nn.Module):
    def __init__(self):
        super(EffiB3, self).__init__()
        self.desc = "[3,224,224] ->EfficientNetB3preTrained[1536] ->256 ->74"
        # Step 1: Load EfficientNet-B3 base model (without pre-trained weights)
        self.base_model = models.efficientnet_b3(weights=None)  # No pretrained weights at first
        self.base_model.classifier = nn.Identity()  # Remove final classification layer
        
        # Step 2: Load the pre-trained weights into the base model
        state_dict = torch.load("preTrained/efficientnet_b3_weights.pth", weights_only=True)
        # Remove classifier weights from state_dict to avoid the mismatch
        state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
        
        # Load the filtered state_dict into the model
        self.base_model.load_state_dict(state_dict)
        # print("EfficientNet-B3 weights loaded successfully.")
        
        # Custom layers added after the base model
        self.batch_norm = nn.BatchNorm1d(1536)  # Output size of EfficientNet-B3 before classifier is 1536
        self.fc1 = nn.Linear(1536, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 74)
        
        # Step 3: Freeze the base model layers if specified
        for param in self.base_model.parameters():
            param.requires_grad = False  # Freeze base model layers
        
    def forward(self, x):
        # Forward pass through EfficientNet-B3 base model
        x = self.base_model(x)  # Get features from EfficientNet-B3
        x = self.batch_norm(x)   # Apply batch normalization
        x = self.fc1(x)          # Apply fully connected layer 1
        x = self.relu(x)         # ReLU activation
        x = self.dropout(x)      # Apply dropout
        x = self.fc2(x)          # Output layer
        return x

class ResN50(nn.Module):
    def __init__(self):
        super(ResN50, self).__init__()

        self.desc = "[3,224,224] -> ResNet50preTrained[2048] -> 256 -> 74"

        # Step 1: Load ResNet50 base model (without pre-trained weights)
        self.base_model = models.resnet50(weights=None)  # No pretrained weights at first
        self.base_model.fc = nn.Identity()  # Remove the final fully connected (fc) layer

        # Step 2: Load the pre-trained weights into the base model
        state_dict = torch.load("preTrained/resnet50_pretrained.pth", weights_only=True)
        self.base_model.load_state_dict(state_dict, strict=False)  # Uncomment if you have custom pretrained weights

        # Custom layers after the base model
        self.fc1 = nn.Linear(2048, 256)  # Updated: Custom fully connected layer for 2048 input features
        self.batch_norm = nn.BatchNorm1d(256)  # Apply BatchNorm after fc1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 74)  # Output layer (num_classes will be 74)

        # Step 3: Freeze the base model layers if specified
        for param in self.base_model.parameters():
                param.requires_grad = False  # Freeze base model layers
        
    def forward(self, x):
        # Forward pass through ResNet50 base model
        x = self.base_model(x)  # Get features from ResNet50
        x = self.fc1(x)          # Apply fully connected layer 1
        x = self.batch_norm(x)   # Apply batch normalization after fc1
        x = self.relu(x)         # ReLU activation
        x = self.dropout(x)      # Apply dropout
        x = self.fc2(x)          # Output layer
        return x

class ResN50_10(nn.Module):
    def __init__(self):
        super(ResN50_10, self).__init__()

        self.desc = "[3,224,224] -> ResNet50preTrained(-10:)[2048] -> 256 -> 74"

        # Step 1: Load ResNet50 base model (without pre-trained weights)
        self.base_model = models.resnet50(weights=None)  # No pretrained weights at first
        self.base_model.fc = nn.Identity()  # Remove the final fully connected (fc) layer

        # Step 2: Load the pre-trained weights into the base model
        state_dict = torch.load("preTrained/resnet50_pretrained.pth", weights_only=True)
        self.base_model.load_state_dict(state_dict, strict=False)  # Load the pre-trained weights

        # Custom layers after the base model
        self.fc1 = nn.Linear(2048, 256)  # Fully connected layer
        self.batch_norm = nn.BatchNorm1d(256)  # Apply BatchNorm after fc1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 74)  # Output layer (num_classes will be 74)

        # Step 3: Create a list of all parameters
        self.params = list(self.base_model.named_parameters())
        # print(f"Total parameters: {len(self.params)}")

        # Identify the last 10 layers
        self.last_10_params = self.params[-10:]  # Get the last 10 parameters

        # Step 4: Freeze all layers except the last 10 layers
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False  # Freeze all layers initially

        # Step 5: Unfreeze the last 10 layers
        for name, param in self.last_10_params:
            param.requires_grad = True  # Unfreeze the last 10 layers

    def forward(self, x):
        # Forward pass through ResNet50 base model
        x = self.base_model(x)  # Get features from ResNet50
        x = self.fc1(x)          # Apply fully connected layer 1
        x = self.batch_norm(x)   # Apply batch normalization after fc1
        x = self.relu(x)         # ReLU activation
        x = self.dropout(x)      # Apply dropout
        x = self.fc2(x)          # Output layer
        return x

class EffiB3_10(nn.Module):
    def __init__(self):
        super(EffiB3_10, self).__init__()
        self.desc = "[3,224,224] ->EfficientNetB3preTrained(-10:)[1536] ->256 ->74"
        # Step 1: Load EfficientNet-B3 base model (without pre-trained weights)
        self.base_model = models.efficientnet_b3(weights=None)  # No pretrained weights at first
        self.base_model.classifier = nn.Identity()  # Remove final classification layer
        
        # Step 2: Load the pre-trained weights into the base model
        state_dict = torch.load("preTrained/efficientnet_b3_weights.pth", weights_only=True)
        # Load the filtered state_dict into the model
        self.base_model.load_state_dict(state_dict,strict=False)
        # print("EfficientNet-B3 weights loaded successfully.")
        
        # Custom layers added after the base model
        self.batch_norm = nn.BatchNorm1d(1536)  # Output size of EfficientNet-B3 before classifier is 1536
        self.fc1 = nn.Linear(1536, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 74)
        
         # Step 3: Create a list of all parameters
        self.params = list(self.base_model.named_parameters())
        # print(f"Total parameters: {len(self.params)}")

        # Identify the last 10 layers
        self.last_10_params = self.params[-10:]  # Get the last 10 parameters

        # Step 4: Freeze all layers except the last 10 layers
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False  # Freeze all layers initially

        # Step 5: Unfreeze the last 10 layers
        for name, param in self.last_10_params:
            param.requires_grad = True  # Unfreeze the last 10 layers
        
    def forward(self, x):
        # Forward pass through EfficientNet-B3 base model
        x = self.base_model(x)  # Get features from EfficientNet-B3
        x = self.batch_norm(x)   # Apply batch normalization
        x = self.fc1(x)          # Apply fully connected layer 1
        x = self.relu(x)         # ReLU activation
        x = self.dropout(x)      # Apply dropout
        x = self.fc2(x)          # Output layer
        return x

class ResN50_15(nn.Module):
    def __init__(self):
        super(ResN50_15, self).__init__()

        self.desc = "[3,224,224] -> ResNet50preTrained(-15:)[2048] -> 256 -> 74"

        # Step 1: Load ResNet50 base model (without pre-trained weights)
        self.base_model = models.resnet50(weights=None)  # No pretrained weights at first
        self.base_model.fc = nn.Identity()  # Remove the final fully connected (fc) layer

        # Step 2: Load the pre-trained weights into the base model
        state_dict = torch.load("preTrained/resnet50_pretrained.pth", weights_only=True)
        self.base_model.load_state_dict(state_dict, strict=False)  # Load the pre-trained weights

        # Custom layers after the base model
        self.fc1 = nn.Linear(2048, 256)  # Fully connected layer
        self.batch_norm = nn.BatchNorm1d(256)  # Apply BatchNorm after fc1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 74)  # Output layer (num_classes will be 74)

        # Step 3: Create a list of all parameters
        self.params = list(self.base_model.named_parameters())
        # print(f"Total parameters: {len(self.params)}")

        # Identify the last 10 layers
        self.last_15_params = self.params[-15:]  # Get the last 10 parameters

        # Step 4: Freeze all layers except the last 10 layers
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False  # Freeze all layers initially

        # Step 5: Unfreeze the last 10 layers
        for name, param in self.last_15_params:
            param.requires_grad = True  # Unfreeze the last 10 layers

    def forward(self, x):
        # Forward pass through ResNet50 base model
        x = self.base_model(x)  # Get features from ResNet50
        x = self.fc1(x)          # Apply fully connected layer 1
        x = self.batch_norm(x)   # Apply batch normalization after fc1
        x = self.relu(x)         # ReLU activation
        x = self.dropout(x)      # Apply dropout
        x = self.fc2(x)          # Output layer
        return x

class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()

        self.desc = "[3,224,224] -> VGG19preTrained[4096] -> 256 -> 74"

        # Step 1: Load VGG19 base model (without pre-trained weights)
        self.base_model = models.vgg19(weights=None)  # Load VGG19 without pre-trained weights
        self.base_model.classifier = nn.Sequential(*list(self.base_model.classifier.children())[:-1])  # Remove the final fully connected (fc) layer

        # Step 2: Load the pre-trained weights into the base model
        state_dict = torch.load("preTrained/vgg19_pretrained.pth", weights_only=True)  # Path to custom pretrained weights
        self.base_model.load_state_dict(state_dict, strict=False)  # Load the state_dict, assuming you have custom weights

        # Custom layers after the base model
        self.fc1 = nn.Linear(4096, 256)  # Custom fully connected layer for 4096 input features (VGG19 FC layer before output)
        self.batch_norm = nn.BatchNorm1d(256)  # Apply BatchNorm after fc1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 74)  # Output layer (num_classes will be 74)

        # Step 3: Freeze the base model layers if specified
        for param in self.base_model.parameters():
            param.requires_grad = False  # Freeze base model layers

    def forward(self, x):
        # Forward pass through VGG19 base model
        x = self.base_model(x)  # Get features from VGG19
        x = self.fc1(x)         # Apply fully connected layer 1
        x = self.batch_norm(x)  # Apply batch normalization after fc1
        x = self.relu(x)        # ReLU activation
        x = self.dropout(x)     # Apply dropout
        x = self.fc2(x)         # Output layer
        return x

class ResN50_20(nn.Module):
    def __init__(self):
        super(ResN50_20, self).__init__()

        self.desc = "[3,224,224] -> ResNet50preTrained(-15:)[2048] -> 256 -> 74"

        # Step 1: Load ResNet50 base model (without pre-trained weights)
        self.base_model = models.resnet50(weights=None)  # No pretrained weights at first
        self.base_model.fc = nn.Identity()  # Remove the final fully connected (fc) layer

        # Step 2: Load the pre-trained weights into the base model
        state_dict = torch.load("preTrained/resnet50_pretrained.pth", weights_only=True)
        self.base_model.load_state_dict(state_dict, strict=False)  # Load the pre-trained weights

        # Custom layers after the base model
        self.fc1 = nn.Linear(2048, 256)  # Fully connected layer
        self.batch_norm = nn.BatchNorm1d(256)  # Apply BatchNorm after fc1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 74)  # Output layer (num_classes will be 74)

        # Step 3: Create a list of all parameters
        self.params = list(self.base_model.named_parameters())
        # print(f"Total parameters: {len(self.params)}")

        # Identify the last 10 layers
        self.last_15_params = self.params[-20:]  # Get the last 10 parameters

        # Step 4: Freeze all layers except the last 10 layers
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False  # Freeze all layers initially

        # Step 5: Unfreeze the last 10 layers
        for name, param in self.last_15_params:
            param.requires_grad = True  # Unfreeze the last 10 layers

    def forward(self, x):
        # Forward pass through ResNet50 base model
        x = self.base_model(x)  # Get features from ResNet50
        x = self.fc1(x)          # Apply fully connected layer 1
        x = self.batch_norm(x)   # Apply batch normalization after fc1
        x = self.relu(x)         # ReLU activation
        x = self.dropout(x)      # Apply dropout
        x = self.fc2(x)          # Output layer
        return x

class ResN50_25(nn.Module):
    def __init__(self):
        super(ResN50_25, self).__init__()

        self.desc = "[3,224,224] -> ResNet50preTrained(-15:)[2048] -> 256 -> 74"

        # Step 1: Load ResNet50 base model (without pre-trained weights)
        self.base_model = models.resnet50(weights=None)  # No pretrained weights at first
        self.base_model.fc = nn.Identity()  # Remove the final fully connected (fc) layer

        # Step 2: Load the pre-trained weights into the base model
        state_dict = torch.load("preTrained/resnet50_pretrained.pth", weights_only=True)
        self.base_model.load_state_dict(state_dict, strict=False)  # Load the pre-trained weights

        # Custom layers after the base model
        self.fc1 = nn.Linear(2048, 256)  # Fully connected layer
        self.batch_norm = nn.BatchNorm1d(256)  # Apply BatchNorm after fc1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 74)  # Output layer (num_classes will be 74)

        # Step 3: Create a list of all parameters
        self.params = list(self.base_model.named_parameters())
        # print(f"Total parameters: {len(self.params)}")

        # Identify the last 10 layers
        self.last_25_params = self.params[-25:]  # Get the last 10 parameters

        # Step 4: Freeze all layers except the last 10 layers
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False  # Freeze all layers initially

        # Step 5: Unfreeze the last 10 layers
        for name, param in self.last_25_params:
            param.requires_grad = True  # Unfreeze the last 10 layers

    def forward(self, x):
        # Forward pass through ResNet50 base model
        x = self.base_model(x)  # Get features from ResNet50
        x = self.fc1(x)          # Apply fully connected layer 1
        x = self.batch_norm(x)   # Apply batch normalization after fc1
        x = self.relu(x)         # ReLU activation
        x = self.dropout(x)      # Apply dropout
        x = self.fc2(x)          # Output layer
        return x


class ResN50_35(nn.Module):
    def __init__(self):
        super(ResN50_35, self).__init__()

        self.desc = "[3,224,224] -> ResNet50preTrained(-35:)[2048] -> 256 -> 74"

        # Step 1: Load ResNet50 base model (without pre-trained weights)
        self.base_model = models.resnet50(weights=None)  # No pretrained weights at first
        self.base_model.fc = nn.Identity()  # Remove the final fully connected (fc) layer

        # Step 2: Load the pre-trained weights into the base model
        state_dict = torch.load("preTrained/resnet50_pretrained.pth", weights_only=True)
        self.base_model.load_state_dict(state_dict, strict=False)  # Load the pre-trained weights

        # Custom layers after the base model
        self.fc1 = nn.Linear(2048, 256)  # Fully connected layer
        self.batch_norm = nn.BatchNorm1d(256)  # Apply BatchNorm after fc1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 74)  # Output layer (num_classes will be 74)

        # Step 3: Create a list of all parameters
        self.params = list(self.base_model.named_parameters())
        # print(f"Total parameters: {len(self.params)}")

        # Identify the last 10 layers
        self.last_35_params = self.params[-35:]  # Get the last 10 parameters

        # Step 4: Freeze all layers except the last 10 layers
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False  # Freeze all layers initially

        # Step 5: Unfreeze the last 10 layers
        for name, param in self.last_35_params:
            param.requires_grad = True  # Unfreeze the last 10 layers

    def forward(self, x):
        # Forward pass through ResNet50 base model
        x = self.base_model(x)  # Get features from ResNet50
        x = self.fc1(x)          # Apply fully connected layer 1
        x = self.batch_norm(x)   # Apply batch normalization after fc1
        x = self.relu(x)         # ReLU activation
        x = self.dropout(x)      # Apply dropout
        x = self.fc2(x)          # Output layer
        return x

class ResN50_un(nn.Module):
    def __init__(self):
        super(ResN50_un, self).__init__()

        self.desc = "[3,224,224] -> ResNet50[2048] -> 256 -> 74"

        # Step 1: Load ResNet50 base model (without pre-trained weights)
        self.base_model = models.resnet50(weights=None)  # No pretrained weights at first
        self.base_model.fc = nn.Identity()  # Remove the final fully connected (fc) layer

        # Custom layers after the base model
        self.fc1 = nn.Linear(2048, 256)  # Fully connected layer
        self.batch_norm = nn.BatchNorm1d(256)  # Apply BatchNorm after fc1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 74)  # Output layer (num_classes will be 74)


    def forward(self, x):
        # Forward pass through ResNet50 base model
        x = self.base_model(x)  # Get features from ResNet50
        x = self.fc1(x)          # Apply fully connected layer 1
        x = self.batch_norm(x)   # Apply batch normalization after fc1
        x = self.relu(x)         # ReLU activation
        x = self.dropout(x)      # Apply dropout
        x = self.fc2(x)          # Output layer
        return x

class ResN50_40(nn.Module):
    def __init__(self):
        super(ResN50_40, self).__init__()

        self.desc = "[3,224,224] -> ResNet50preTrained(-40:)[2048] -> 256 -> 74"

        # Step 1: Load ResNet50 base model (without pre-trained weights)
        self.base_model = models.resnet50(weights=None)  # No pretrained weights at first
        self.base_model.fc = nn.Identity()  # Remove the final fully connected (fc) layer

        # Step 2: Load the pre-trained weights into the base model
        state_dict = torch.load("preTrained/resnet50_pretrained.pth", weights_only=True)
        self.base_model.load_state_dict(state_dict, strict=False)  # Load the pre-trained weights

        # Custom layers after the base model
        self.fc1 = nn.Linear(2048, 256)  # Fully connected layer
        self.batch_norm = nn.BatchNorm1d(256)  # Apply BatchNorm after fc1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 74)  # Output layer (num_classes will be 74)

        # Step 3: Create a list of all parameters
        self.params = list(self.base_model.named_parameters())
        # print(f"Total parameters: {len(self.params)}")

        # Identify the last 10 layers
        self.last_40_params = self.params[-40:]  # Get the last 10 parameters

        # Step 4: Freeze all layers except the last 10 layers
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False  # Freeze all layers initially

        # Step 5: Unfreeze the last 10 layers
        for name, param in self.last_40_params:
            param.requires_grad = True  # Unfreeze the last 10 layers

    def forward(self, x):
        # Forward pass through ResNet50 base model
        x = self.base_model(x)  # Get features from ResNet50
        x = self.fc1(x)          # Apply fully connected layer 1
        x = self.batch_norm(x)   # Apply batch normalization after fc1
        x = self.relu(x)         # ReLU activation
        x = self.dropout(x)      # Apply dropout
        x = self.fc2(x)          # Output layer
        return x



