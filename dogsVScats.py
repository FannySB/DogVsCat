from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets as Dataset
import torch
# from torch.optim import zero_grad
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

data_dir_train = 'dataset/trainset/'
data_dir_test = 'dataset/testset/'
transforms = transforms.Compose([transforms.Resize(128),
                                 transforms.CenterCrop(128),
                                 transforms.ToTensor()]
                                )
train_data = Dataset.ImageFolder(data_dir_train, transform=transforms)
split_train_data, split_valid_data = Dataset.random_spit(train_data, [len(train_data)*0.80, len(train_data)*0.20])

test_data = Dataset.ImageFolder(data_dir_test, transform=transforms)

# train_loader_cat = DataLoader(train_data_cat, batch_size=32, shuffle=True)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# train_loader_dog = DataLoader(train_data_dog, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()


# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.sigmoid(x)
        return x


model = Net()
print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        with torch.no_grad():
            m.bias.zero_()
        # nn.init.zeros_(m.bias)


model.apply(weight_init)
learning_rate = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# criterion = nn.CrossEntropyLoss().cuda()
criterion = nn.BCELoss().cuda()

n_epochs = 5  # you may increase this number to train a final model
valid_loss_min = np.Inf  # track change in validation loss
from torch.autograd import Variable

for epoch in range(1, n_epochs + 1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    accuracy = 0.0
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        target = target.float()

        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item() * data.size(0)
    # print(((output.squeeze() > 0.5) == target.byte()).sum().item() / target.shape[0])

    ######################
    # validate the model #
    ######################
    model.eval()

    for data, target in test_loader:

        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        target = target.float()
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target.view_as(output))
        # update average validation loss
        valid_loss += loss.item() * data.size(0)

        t = Variable(torch.FloatTensor([0.5]))  # threshold
        out = (output > t.cuda(async=True)).float() * 1
        # print(out)

        equals = target.float() == out.t()
        # print(equals)
        # print(torch.sum(equals))
        accuracy += (torch.sum(equals).cpu().numpy())
    # print(equals)
    # print(target)

    # calculate average losses
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(test_loader.dataset)
    accuracy = accuracy / len(test_loader.dataset)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}  \tAccuracy: {:.6f}  '.format(
        epoch, train_loss, valid_loss, accuracy))

    # save model if validation loss has decreased
    if valid_loss < valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(), 'my:model.pt')
        valid_loss_min = valid_loss