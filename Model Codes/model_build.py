#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

'''
Medical Plant Image Classification Project.
Author: Binu raj
Created on: 15 March 2024
Updated by: shabeen abdul varis
Updated on: 29 August 2024
'''

# Configuration flags for training pipeline.
custom_data_transformation = True    # Flag to indicate whether to use custom image transformation(True) or default vit transformation.
log_neptune = False                  # Flag to enable/disable logging of training metrics using Neptune.
whole_data_train = True              # Flag to indicate if the model should be trained on entire data(both training and testing data) (Model Finalization).   

# Hyperparameters.
path = "/home/shabeen/data/Dataset/"     # Root directory for image dataset
epoch = 30                               # Number of training epochs
optimizer_name = 'adam'                  # Optimizer type
batch_size = 32                          # Number of images per training batch
lr = 5e-5                                # Learning rate for model optimization

# Model name for saving the trained model.
model_used = "vit"

# Import required libraries for deep learning, data processing, and utilities.
import neptune
import os , copy
import torch
from tqdm.auto import tqdm
import torchvision
from typing import List, Dict, Tuple
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import timm
from torchinfo import summary

# Import PIL to resolve image truncated error.
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Generate timestamp for unique model and experiment.
from datetime import datetime
now = datetime.now()
date = now.strftime('%b%d')

# List of image classes (folders) in the dataset.
image_classes = os.listdir(path)
total_class = len(image_classes)

# Generate a unique name for the Neptune logging.
neptune_run_name = f"{model_used}_{date}_{epoch}epch_{batch_size}bsze"

# Function to set random seeds for reproducibility.
def set_seeds(seed: int=42):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

# Use the GPU for training, or the CPU if GPU is not available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device,"|",torch.cuda.get_device_name(0))

# Function to connect to Neptune for logging.
def connect_neptune() :
	run = neptune.init_run(
	project = "medicinalplants/plants-identifier",
	api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NzYzMmJkYi04NTU4LTQwMjYtYWEzZC1hNjJkYTExMDdmZjkifQ==",
	name = neptune_run_name)

	run["parameters"] ={
	"learning_rate": lr,
	"epochs": epoch,
	"batch size": batch_size,
	"optimizer":optimizer_name
	}
	
	return run

		
# Get pretrained weights for ViT-Base model.(for getting default vit transformation)
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

# Function to load pretrained model weights.
def update_model_weights(model : torch.nn.Module,
		filename : str ):
		
	if not os.path.exists(filename):
		raise FileNotFoundError

	# Load model state dictionary.
	model_state = torch.load(filename, map_location=device)
	model.load_state_dict(model_state['model'])

# Pretrained model weights path.
pretrained_model_weights = 'Pretrained_weight_(plantnet)/vit_base_patch16_224_weights_best_acc.tar' 

# Initializing a pretrained vit model from timm library.
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1081)
model.to(device)   # Move model to computation device

# Load pretrained weights.
update_model_weights(model, filename = pretrained_model_weights)

# Freeze all layers in the model so that their weights are not updated during training.
for parameter in model.parameters():
	parameter.requires_grad = False

# Change the classifier head (final layer) to match the number of classes in your dataset.
set_seeds() # Set seeds for reproducibility.
model.head = nn.Linear(in_features=768, out_features=total_class).to(device)

# Display model architecture and number of trainable parameters.
summary(model = model,
        input_size = (32, 3, 224, 224),
        col_names = ["input_size", "output_size", "num_params", "trainable"],
        col_width = 20,
        row_settings = ["var_names"]
       )

# Define image transformations for training and testing.
if custom_data_transformation:
    
	train_transforms = transforms.Compose([
        transforms.Lambda(lambda img: transforms.RandomResizedCrop(264,scale=(0.5,1))(img) if img.width>500 else transforms.Resize((232,232))(img)),
		transforms.RandomRotation(180),
		transforms.CenterCrop(224), 
		transforms.RandomHorizontalFlip(0.7),
		transforms.RandomVerticalFlip(p=0.6),
		# transforms.RandomGrayscale(p=0.2),
		# transforms.GaussianBlur(kernel_size=3),
		transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25,hue=0.01), 
		transforms.ToTensor(), 
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) # Normalize
	
	test_transforms = transforms.Compose([
		transforms.RandomResizedCrop(300,scale=(0.9,1)),
		#transforms.RandomRotation(180),
		transforms.CenterCrop(224),
		transforms.RandomHorizontalFlip(0.4),
		transforms.RandomHorizontalFlip(0.4),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	    
	print("\nData transformation |  \ntrain : ",train_transforms,"\ntest : ",test_transforms,"\n")
	final_transform = [train_transforms,test_transforms]

else:
	# Use default pretrained ViT transformations.
	pretrained_vit_transforms = pretrained_vit_weights.transforms()
    
	print("\nData transformation |  \n",pretrained_vit_transforms,"\n")
	final_transform = pretrained_vit_transforms

NUM_WORKERS = 12    # Number of CPU cores for data loading.

# Function to create dataloaders for training and testing.
# data pipeline parts
def create_dataloaders(
		path : str,
		transform ,
		batch_size: int,
		num_workers: int = NUM_WORKERS
		) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

	# If transformations are provided as a list (for train and test separately).
	if isinstance(transform, list):
		train_transform, test_transform = transform  

	# Apply same transformation for both train and test.
	else :
		train_transform = test_transform = transform

	# Custom dataset class for loading images (for applying different transformation for test and train data pipeline)
	class CustomImageDataset(Dataset):
		def __init__(self, image_dir, transform = None):
			self.image_dir = image_dir
			self.transform = transform
			self.classes = sorted(os.listdir(image_dir))
			self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

			self.image_paths = []
			self.image_labels = []

		def initialise(self):
			for class_name in self.classes:
				class_dir = os.path.join(self.image_dir, class_name)
				if os.path.isdir(class_dir):
					for img_name in os.listdir(class_dir):
						img_path = os.path.join(class_dir, img_name)
						if img_path.endswith(('.jpg', '.png', '.jpeg')):  # Add supported extensions
							self.image_paths.append(img_path)
							self.image_labels.append(self.class_to_idx[class_name])

		def __len__(self):
			return len(self.image_paths)
		    
		def __getitem__(self,idx):
			image = self.image_paths[idx]
			label = self.image_labels[idx]
			image = Image.open(image)
			if self.transform:
				image = self.transform(image)
			return image, label   

	# Train-test split.  
	if not whole_data_train:
		dataset = CustomImageDataset(image_dir = path, transform = None) 
		dataset.initialise()
		class_names = dataset.classes
		train_image_paths,test_image_paths,train_image_labels,test_image_labels = train_test_split(dataset.image_paths,dataset.image_labels,train_size=.8)
	    	
		train_dataset = CustomImageDataset(image_dir = path, transform = train_transform)
		train_dataset.image_paths, train_dataset.image_labels = train_image_paths, train_image_labels
		test_dataset = CustomImageDataset(image_dir = path, transform = test_transform)       
		test_dataset.image_paths, test_dataset.image_labels = test_image_paths, test_image_labels
    
	# Use whole data without splitting.
	else:    
		train_dataset = datasets.ImageFolder(root=path, transform = train_transform) 
		class_names = train_dataset.classes
	
	# Display image data numbers.		
	print(f'\n{len(class_names)} Classes : {class_names}\n')
	print(f"total data : {len(dataset)}\ntrain data : {len(train_dataset)}\ntest data : {len(test_dataset)}\n") if not whole_data_train else print(f"total training data : {len(train_dataset)}\n")
	
	# Create DataLoader for train and test datasets
	train_dataloader = DataLoader(
		train_dataset,
		batch_size = batch_size,
		shuffle = True,
		num_workers = num_workers,
		pin_memory = True
    )
		
	test_dataloader = DataLoader(
		test_dataset,
		batch_size = batch_size,
		shuffle = False,
		num_workers = num_workers,
		pin_memory = True
    ) if not whole_data_train else None
        
	return train_dataloader, test_dataloader

# Training step function to process a single epoch
def train_step(model : torch.nn.Module, 
		dataloader : torch.utils.data.DataLoader, 
		loss_fn : torch.nn.Module, 
		optimizer : torch.optim.Optimizer,
		device : torch.device
		) -> Tuple[float, float]:
		
	model.train()
	train_loss, train_acc = 0, 0
	
	for batch, (X, y) in enumerate(dataloader):
		X, y = X.to(device), y.to(device)
		optimizer.zero_grad()
		y_pred = model(X)
		loss = loss_fn(y_pred, y)
		train_loss += loss.item()
		loss.backward()
		optimizer.step()

		y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
		train_acc += (y_pred_class == y).sum().item()/len(y_pred)

	train_loss = train_loss / len(dataloader)
	train_acc = train_acc / len(dataloader)
	
	return train_loss, train_acc

    
# Evaluation step function to assess model performance.
def test_step(model: torch.nn.Module,
		dataloader: torch.utils.data.DataLoader,
		loss_fn: torch.nn.Module,
		device: torch.device
		) -> Tuple[float, float]:

	model.eval()
	test_loss, test_acc = 0, 0
	
	with torch.inference_mode():
		for batch, (X, y) in enumerate(dataloader):
			X, y = X.to(device), y.to(device)
			test_pred_logits = model(X)
			loss = loss_fn(test_pred_logits, y)
			test_loss += loss.item()
			test_pred_labels = test_pred_logits.argmax(dim=1)
			test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
	test_loss = test_loss / len(dataloader)
	test_acc = test_acc / len(dataloader)
	
	return test_loss, test_acc

# Main training engine with early stopping and model checkpointing.
def enginetrain(model: torch.nn.Module,
		train_dataloader:torch.utils.data.DataLoader,
		test_dataloader:torch.utils.data.DataLoader,
		optimizer:torch.optim.Optimizer,
		loss_fn:torch.nn.Module,
		lr_scheduler,
		epochs: int,
		device:torch.device
		) -> Tuple[dict, int]:
		
	model.to(device)
	best_val_loss = float('inf')
	best_model_state=torch.tensor([1],device=device)
	patience=2
	patience_counter = 0
	
	try:
		for epoch in tqdm(range(epochs)):
			# Perform training and testing steps.
			train_loss, train_acc = train_step(model = model, dataloader = train_dataloader,
                                               loss_fn = loss_fn, optimizer = optimizer,
                                               device=device
                                              )
			# Conditional test step based on training mode.
			test_loss, test_acc = test_step(model = model, dataloader = test_dataloader,
                                            loss_fn = loss_fn, device = device
                                           ) if not whole_data_train else (0,0)

            		# Initialize Neptune run on first epoch.
			if epoch==0:
				run=connect_neptune() if log_neptune else False

            		# Update learning rate.
			lr_scheduler.step()

            		# Print epoch metrics.
			print(
			f"Epoch: {epoch+1} | "
			f"train_loss: {train_loss:.4f} | "
			f"train_acc: {train_acc:.4f} | " 
			f"test_loss: {test_loss:.4f} | "
			f"test_acc: {test_acc:.4f} | "
            		f"current_lr:{lr_scheduler.get_last_lr()[0]}" )

            		# Log metrics to Neptune.
			if run:
				run["train/accuracy"].append(train_acc)
				run["test/accuracy"].append(test_acc)
				run["train/loss"].append(train_loss)
				run["test/loss"].append(test_loss)

            		# Determine current loss based on training mode.
			current_loss=train_loss if whole_data_train else test_loss

            		# Early stopping and model checkpointing.
			if current_loss <= best_val_loss:
				best_val_loss = current_loss
				patience_counter = 0 
				best_model_state = copy.deepcopy(model.state_dict())

			else:
				patience_counter += 1
				if patience_counter >= patience:
					print("Early stopping triggered.")
					print("Best validation loss :",best_val_loss)
					print(f"Saving the model parameters trained at epoch-{epoch-1}" )
					epoch=epoch-2
					break

	except Exception as e:
		print(f"An error occurred: {e}")
	finally:
		if run :
			run.stop()
		return best_model_state , epoch+1
		# closimg neptune connection.
		


# Create dataloaders with specified transformations.
train_dataloader_pretrained, test_dataloader_pretrained = create_dataloaders(path=path,
										transform=final_transform,
										batch_size=batch_size
										) 	

# Configure optimizer with weight decay for regularization.
optimizer = torch.optim.Adam(
            params = filter(lambda p:p.requires_grad,model.parameters()),
            lr = lr, weight_decay=1e-3
            )

# Learning rate scheduler to reduce learning rate at specific epochs(milestones).
step_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
		optimizer, milestones = [9,20], gamma = 0.8
		)

# Loss function for multi-class classification.
loss_fn = torch.nn.CrossEntropyLoss()
set_seeds()

# Train the model and retrieve best model state.
model_state , epoch  = enginetrain(model = model, train_dataloader = train_dataloader_pretrained,
				     test_dataloader = test_dataloader_pretrained, optimizer = optimizer,
				     loss_fn = loss_fn, lr_scheduler = step_lr_scheduler,
				     epochs = epoch, device = device
				     )

# Load the best model state.
model.load_state_dict(model_state)

# Save the trained model.
model_name=f"{model_used}_{total_class}cls_{epoch}epch_{date}_{optimizer_name}_{batch_size}bsze.pt"
torch.save(model, f"/home/shabeen/data/{model_name}")
print(f"model saved - {model_name}")
