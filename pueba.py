import torch
import torch.nn as nn
import torchvision.models as models

# 1. Load pretrained ResNet-50 model
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

# 2. (Optional) Freeze early layers if you only want to train the classifier head
# for param in model.parameters():
#     param.requires_grad = False

# 3. Replace the final fully connected layer (fc)
# ResNet-50's original fc layer has 2048 input features
num_features = model.fc.in_features
num_classes = 10  # Change this to your custom dataset class count
model.fc = nn.Linear(num_features, num_classes)

# 4. Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"ResNet-50 modified for {num_classes} output classes.")
