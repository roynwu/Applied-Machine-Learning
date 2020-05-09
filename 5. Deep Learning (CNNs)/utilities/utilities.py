# -*- coding: utf-8 -*-

import torch
import torchvision

# import torch.utils.tensorboard as tb

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

LABEL_NAMES = {'background':0, 'kart':1, 'pickup':2, 'nitro':3, 'bomb':4, 'projectile':5}

LABEL_=['background','kart','pickup','nitro','bomb','projectile']

"""## Reading Data"""

# from google.colab import drive
# drive.mount('/content/drive')

# # upload the data and unzip it. You will see data/ with train/ and valid/. 
# !unzip supertux_classification_trainval.zip

"""## Defining Torch Dataset"""

# # Create data set
# def create_data(input):
#     # Create list that stores path of images
#     import os
#     images_path = []
#     filelist = os.listdir(input)
#     filelist.remove('labels.csv')
#     filelist = sorted(filelist,key=lambda x: int(os.path.splitext(x)[0]))
#     for filename in filelist:
#       images_path.append(input + '/' + filename)
      
#     # Read images and store in list
#     import numpy as np
#     data = []
#     for path in images_path:
#       img = Image.open(path)
#       img = np.array(img)/255
#       img = img.transpose((2, 0, 1))
#       data.append(img)

#     # Covert to Tensor
#     X = torch.Tensor(np.array(data))

#     return X

# Create data set
def create_data(input):
    # Create list that stores path of images
    import os
    images_path = []
    filelist = os.listdir(input)
    filelist.remove('labels.csv')
    filelist = sorted(filelist,key=lambda x: int(os.path.splitext(x)[0]))
    for filename in filelist:
      images_path.append(input + '/' + filename)
      
    # # Read images and store in list
    # import numpy as np
    # data = []
    # for path in images_path:
    #   img = Image.open(path)
    #   img = np.array(img)/255
    #   img = img.transpose((2, 0, 1))
    #   data.append(img)

    # # Covert to Tensor
    # X = torch.Tensor(np.array(data))

    return images_path

def create_label(input):
    # Use pandas to read csv files
    import pandas as pd
    path = input + '/labels.csv'
    df = pd.read_csv(path)
    labels_df = df['label']

    # Use dictionary as lookup table
    import numpy as np
    labels = []
    for label in labels_df:
      labels.append(LABEL_NAMES.get(label))

    # Convert to Tensor
    y = torch.LongTensor(np.array(labels))  

    return y

# class SuperTuxDataset(Dataset):
#     def __init__(self, image_path, data_transforms=None):
#         """
#         Your code here
#         Hint: Use the python csv library to parse labels.csv
#         """
#         # raise NotImplementedError('SuperTuxDataset.__init__')
#         self.X = create_data(image_path)
#         self.y = create_label(image_path)

#     def __len__(self):
#         """
#         Your code here
#         """
#         # raise NotImplementedError('SuperTuxDataset.__len__')
#         return len(self.X)

#     def __getitem__(self, idx):
#         """
#         Your code here
#         return a tuple: img, label
#         """
#         # raise NotImplementedError('SuperTuxDataset.__getitem__')
#         img = self.X[idx]
#         label = self.y[idx]

#         return img, label

class SuperTuxDataset(Dataset):
    def __init__(self, image_path, data_transforms=None):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv
        """
        # raise NotImplementedError('SuperTuxDataset.__init__')
        self.img_paths = create_data(image_path)
        self.y = create_label(image_path)

    def __len__(self):
        """
        Your code here
        """
        # raise NotImplementedError('SuperTuxDataset.__len__')
        return len(self.y)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        # raise NotImplementedError('SuperTuxDataset.__getitem__')
        # img = self.X[idx]
        label = self.y[idx]

        import numpy as np
        img = Image.open(self.img_paths[idx])
        img = np.array(img)/255
        img = img.transpose((2, 0, 1))

        # Covert to Tensor
        img = torch.Tensor(np.array(img))
        

        return img, label

"""The following utility visualizes the data, optionally, as a sanity check for your implementation of the dataset class. Call visualize_data() after setting the correct variables inside this code snippet."""

# def visualize_data():

#     Path_to_your_data = '/content/data/train'
#     dataset = SuperTuxDataset(image_path=Path_to_your_data, data_transforms=transforms.ToTensor())

#     f, axes = plt.subplots(3, len(LABEL_NAMES))

#     counts = [0]*len(LABEL_NAMES)

#     for img, label in dataset:
#         c = counts[label]

#         if c < 3:
#             ax = axes[c][label]
#             ax.imshow(img.permute(1, 2, 0).numpy())
#             ax.axis('off')
#             ax.set_title(LABEL_[label])
#             counts[label] += 1
        
#         if sum(counts) >= 3 * len(LABEL_NAMES):
#             break

#     plt.show()

# visualize_data()

"""## Defining Model Architecture and Loss"""

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here
        Compute mean(-log(softmax(input)_label))
        @input:  torch.Tensor((B,C)), where B = batch size, C = number of classes
        @target: torch.Tensor((B,), dtype=torch.int64)
        @return:  torch.Tensor((,))
        Hint: Don't be too fancy, this is a one-liner
        """
        # raise NotImplementedError('ClassificationLoss.forward')
        # return F.cross_entropy(input, target)

        # num_examples = target.shape[0]
        # batch_size = input.shape[0]
        # output = F.log_softmax(input)
        # output = output[range(batch_size), target]

        # return - torch.sum(output)/num_examples

        output = F.log_softmax(input, dim=1)
        output = output[range(input.shape[0]), target]

        return - torch.mean(output)
        

# class CNNClassifier(torch.nn.Module):
#     def __init__(self):
#         """
#         Your code here
#         """
#         # raise NotImplementedError('CNNClassifier.__init__')
#         super(CNNClassifier, self).__init__() 
#         self.features = nn.Sequential(nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),  
#                                       nn.ReLU(inplace=True),
#                                       nn.MaxPool2d(kernel_size=2, stride=2),
#                                       nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
#                                       nn.ReLU(inplace=True),
#                                       nn.MaxPool2d(kernel_size=2, stride=2),
#                                       nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2),
#                                       nn.ReLU(inplace=True),
#                                       nn.MaxPool2d(kernel_size=2, stride=2))
#         self.classifier = nn.Sequential(nn.Linear(64, 64),
#                                       nn.ReLU(inplace=True),
#                                       nn.Linear(64, 32),
#                                       nn.ReLU(inplace=True),
#                                       nn.Linear(32, 6))

class CNNClassifier(torch.nn.Module):
    def __init__(self):
        """
        Your code here
        """
        # raise NotImplementedError('CNNClassifier.__init__')
        super(CNNClassifier, self).__init__() 
        self.features = nn.Sequential(nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),  
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2))
                                      # nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2),
                                      # nn.ReLU(inplace=True),
                                      # nn.MaxPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Sequential(nn.Linear(512, 128),
                                      nn.ReLU(inplace=True),
                                      # nn.Linear(64, 32),
                                      # nn.ReLU(inplace=True),
                                      nn.Linear(128, 6))


    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        # raise NotImplementedError('CNNClassifier.forward')
        x = self.features(x)
        x = x.view(x.size(0),-1)
        out = self.classifier(x)
    
        return out

# from torch import save
# from torch import load
# from os import path

# def save_model(model):
#     if isinstance(model, CNNClassifier):
#         # return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
#         # return save(model.state_dict(), path.join(path.dirname(path.abspath('')), 'cnn.th'))
#         return save(model.state_dict(), 'cnn.th')
    
#     raise ValueError("model type '%s' not supported!"%str(type(model)))


# def load_model():
#     r = CNNClassifier()
#     # r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
#     # r.load_state_dict(load(path.join(path.dirname(path.abspath('')), 'cnn.th'), map_location='cpu'))
#     r.load_state_dict(load('cnn.th'))
#     return r

"""## Tensorboard logging"""

def test_logging(train_logger, valid_logger):

    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epoch
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """

    # This is a strongly simplified training loop
    train_loss_step = 0
    train_acc_step = 0
    valid_acc_step = 0
    for epoch in range(10):
        dummy_train_accuracy_lst = []
        dummy_validation_accuracy_lst = []
        torch.manual_seed(epoch)
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = epoch/10. + torch.randn(10)
            # raise NotImplementedError('Log the training loss')
            train_logger.add_scalar('loss', dummy_train_loss, train_loss_step)
            train_loss_step += 1
            dummy_train_accuracy_lst.append(dummy_train_accuracy.mean())
        # raise NotImplementedError('Log the training accuracy')
        train_logger.add_scalar('accuracy', sum(dummy_train_accuracy_lst) / 
                                len(dummy_train_accuracy_lst), train_acc_step)
        train_acc_step += 1
        torch.manual_seed(epoch)
        for iteration in range(10):
            dummy_validation_accuracy = epoch/10. + torch.randn(10)
            dummy_validation_accuracy_lst.append(dummy_validation_accuracy.mean())
        # raise NotImplementedError('Log the validation accuracy')
        valid_logger.add_scalar('accuracy', sum(dummy_validation_accuracy_lst) / 
                                len(dummy_validation_accuracy_lst), valid_acc_step)
        valid_acc_step += 1

"""After implementing `test_logging()`, call it below. This should produce some plots on your tensorboard."""

# %load_ext tensorboard

# %reload_ext tensorboard

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# from torch.utils.tensorboard import SummaryWriter
# ROOT_LOG_DIR = './logdir'

# %tensorboard --logdir {ROOT_LOG_DIR} #Launch tensorboard

# train_logger = tb.SummaryWriter(path.join('./logdir', 'train'))
# valid_logger = tb.SummaryWriter(path.join('./logdir', 'test'))
# test_logging(train_logger, valid_logger)

"""**Training and evaluation utility functions** 

Here are some implementations of useful functions for training and evaluating your models. Read these carefully. You may need to make some obvious edits before these will work.
"""

# def accuracy(outputs, labels):
#     outputs_idx = outputs.max(1)[1].type_as(labels)
#     return outputs_idx.eq(labels).float().mean()

# def predict(model, inputs, device='cpu'):
#     inputs = inputs.to(device)
#     logits = model(inputs)
#     return F.softmax(logits, -1)

# def draw_bar(axis, preds, labels=None):
#     y_pos = np.arange(6)
#     axis.barh(y_pos, preds, align='center', alpha=0.5)
#     axis.set_xticks(np.linspace(0, 1, 10))
    
#     if labels:
#         axis.set_yticks(y_pos)
#         axis.set_yticklabels(labels)
#     else:
#         axis.get_yaxis().set_visible(False)
    
#     axis.get_xaxis().set_visible(False)

# def visualize_predictions():
#     from torchvision.transforms import functional as TF
  
#     model = load_model()
#     model.eval()

#     validation_image_path='/content/data/valid' #enter the path 

#     dataset = SuperTuxDataset(image_path=validation_image_path)

#     f, axes = plt.subplots(2, 6)

#     idxes = np.random.randint(0, len(dataset), size=6)

#     for i, idx in enumerate(idxes):
#         img, label = dataset[idx]
#         preds = predict(model, img[None], device='cpu').detach().cpu().numpy()

#         # axes[0, i].imshow(TF.to_pil_image(img))
#         axes[0, i].imshow(img.permute(1, 2, 0).numpy())
#         axes[0, i].axis('off')
#         draw_bar(axes[1, i], preds[0], LABEL_ if i == 0 else None)

#     plt.show()

"""## Training models

The `load_data` utility below uses your implementation of the dataset class above to provide a helper function that might be useful when you train your models. You won't need to change anything inside this function.
"""

# def load_data(dataset_path, data_transforms=None, num_workers=0, batch_size=128):
#     dataset = SuperTuxDataset(dataset_path,data_transforms)
#     return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)

"""But you *will* need to implement `train()`, which takes an `args` object, that could have arbitrary arguments inside. We won't test your train function directly, but will instead evaluate the model it produces as output. To call `train`, you have to first create an args object, and add various attributes to it, as shown below:"""

# class Args(object):
#   pass

# args = Args();
# # Add attributes to args here, such as:
# # args.log_dir = './my_tensorboard_log_directory' 
# args.epochs = 20
# args.lr = 0.005
# args.momentum = 0.9

"""Then implement `train`. Follow the instructions in the assignment."""

# def train(args):
#     """
#     Your code here
#     """
#     # Create dataloader, model, loss, optimizer
#     train_dataloader = load_data('/content/data/train')
#     model = CNNClassifier()
#     loss = ClassificationLoss()
#     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

#     # Create SummaryWriter
#     # if args.log_dir is not None:
#     #     train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
#     #     valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

#     # Set overall step
#     overall_step = 0

#     # Send model to device
#     model.to(device)

#     # Training loop
#     for i in range(args.epochs):
#       # Keep track of loss and accuracy
#       total_loss = 0
#       running_loss = 0
#       correct = 0
#       total = 0 

#       # Enumeration loop of dataloader
#       for idx, (X,y) in enumerate(train_dataloader):
#         # Send to device
#         X, y = X.to(device), y.to(device)
#         # Zero out gradients of optimizer
#         optimizer.zero_grad()
#         # Input training into model
#         y_pred = model.forward(X)
#         # Calculate loss
#         train_loss = loss(y_pred,y)
#         # Backpropagate
#         train_loss.backward()
#         # Step the optimizer
#         optimizer.step()
#         # Update total/running loss
#         total_loss += train_loss.item()
#         running_loss += train_loss.item()
#         # Update correct
#         _, predicted = torch.max(y_pred.data, 1)
#         total += y.size(0)
#         correct += (predicted == y).sum().item()

#         # Print every 20 mini-batches and add to Tensorboard
#         if idx % 20 == 19:
#           print('[%d, %2d] loss: %.3f' % (i + 1, idx + 1, running_loss / 20))
#           # train_logger.add_scalar('Training Loss', running_loss / 20, overall_step)
#           overall_step += 1
#           running_loss = 0

#       # Add accuracy to Tensorboard
#       # train_logger.add_scalar('Training Accuracy', correct / total, i)

#       # Print loss and accuracy after every epoch
#       print('-----')
#       print('Epoch: ' + str(i + 1))
#       print('Loss: ' + str(total_loss / len(train_dataloader)))
#       print('Accuracy: ' + str(100 * correct / total))
#       print('-----')

#     # Save model
#     trained_model = save_model(model)

#     return trained_model

#     # raise NotImplementedError('train')

# def test(net):
#     # Keep track of loss and accuracy
#     running_loss = 0
#     correct = 0
#     total = 0

#     # Create dataloader, loss
#     valid_dataloader = load_data('/content/data/valid')
#     loss = ClassificationLoss()

#     # Set overall step
#     overall_step = 0

#     # Send to device
#     net.to(device)
#     # Eval mode
#     net.eval()

#     # Test loop
#     with torch.no_grad():

#       # Enumeration loop of dataloader
#       for idx, (X,y) in enumerate(valid_dataloader):       
#         # Send to device
#         X, y = X.to(device), y.to(device)
#         # Input data into model
#         y_pred = net.forward(X)
#         # Calculate loss
#         valid_loss = loss(y_pred,y)
#         # Update total/running loss
#         running_loss += valid_loss.item()
#         # Calcuate correct
#         _, predicted = torch.max(y_pred.data, 1)
#         total += y.size(0)
#         correct += (predicted == y).sum().item()

#         # Print every 5 mini-batches and add to Tensorboard
#         if idx % 5 == 4:
#           print('[%2d] loss: %.3f' % (idx + 1, running_loss / 5))
#           # valid_logger.add_scalar('Validation Loss', running_loss / 5, overall_step)
#           overall_step += 1
#           running_loss = 0

#     # Print final accuracy
#     print('Accuracy of the network on the test images: ' + str(
#       100 * correct / total))

"""Now, you can call `train` with `train(args)`, where `args` contains your various favorite settings of hyperparameters and other arguments that your implementation of `train` needs.


Afterwards, you can call `predict()' and `visualize_predictions()' to evaluate your model.
"""

# import torch
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Train attempt 1
# # Train model
# trained_model = train(args)

# # Train attempt 2
# # Train model
# trained_model = train(args)

# # Train attempt 3
# # Train model
# trained_model = train(args)

# # Train attempt 4
# # Train model
# trained_model = train(args)

# # Test attempt 1
# # Load trained model
# test_model = load_model()
# # Test model on validation set
# test(test_model)

# # Test attempt 2
# # Load trained model
# test_model = load_model()
# # Test model on validation set
# test(test_model)

# # Test attempt 3
# # Load trained model
# test_model = load_model()
# # Test model on validation set
# test(test_model)

# # Test attempt 4
# # Load trained model
# test_model = load_model()
# # Test model on validation set
# test(test_model)

# # Visualize predictions attempt 1
# visualize_predictions()

# # Visualize predictions attempt 2
# visualize_predictions()

# # Visualize predictions attempt 3
# visualize_predictions()

