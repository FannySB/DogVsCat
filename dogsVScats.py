from torchvision import transforms
from torch.utils.data import DataLoader
# from torch.utils import data
from torchvision import datasets as Dataset
import torch

# from torch.optim import zero_grad
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from itertools import accumulate
import matplotlib.pyplot as plt

class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths):
    assert sum(lengths) == len(dataset)
    print("Length :", lengths)
    indices = torch.randperm(len(dataset))
    return [Subset(dataset, indices[offset - length:offset])
            for offset, length in zip(accumulate(lengths), lengths)]


data_dir_train = 'dataset/trainset/'
data_dir_test = 'dataset/testset/'

# transforms = transforms.Compose([transforms.Resize(128),
#                                  transforms.CenterCrop(128),
#                                  transforms.ToTensor()]
#                                 )

# transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
#                                  transforms.ToTensor()])

transforms = transforms.Compose([transforms.Resize(128),
                                 transforms.CenterCrop(128),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                                ])
                                #  transforms.RandomResizedCrop(128),
                                
                                
                                #  transforms.Grayscale(num_output_channels=3),
                                #  transforms.Normalize((0,0,0), (1,1,1))
                                # ])

train_data = Dataset.ImageFolder(data_dir_train, transform=transforms)
split_train_data, split_valid_data = random_split(train_data, [int(len(train_data)*0.80), len(train_data)-int(len(train_data)*0.80)])


train_loader = DataLoader(split_train_data, batch_size=32, shuffle=True, drop_last=True)
valid_loader = DataLoader(split_valid_data, batch_size=32, shuffle=True, drop_last=True)


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
        # self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4 * 4 * 4)
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
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
learning_rate = 10e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss().cuda()

n_epochs = 45  # you may increase this number to train a final model
valid_loss_min = np.Inf  # track change in validation loss
from torch.autograd import Variable

train_loss_hist = []
valid_loss_hist = []

scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
for epoch in range(1, n_epochs + 1):

    scheduler.step()

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
        # print("Shape data:", data.size())
        # print("Input = ", output.size())
        # print("Target = ", target.size())
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

    for data, target in valid_loader:

        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        target = target.float()
        output = model(data)

        # print("output ", output)

        # calculate the batch loss
        loss = criterion(output, target.view_as(output))
        # update average validation loss
        valid_loss += loss.item() * data.size(0)

        t = Variable(torch.FloatTensor([0.5]))  # threshold
        out = (output > t.cuda(async=True)).float() * 1
        # print(out)

        # print("out ", out.t())
        # print("target ", target)
        equals = target.float() == out.t()
        # print("equals ", equals)
        # print(equals)
        # print(torch.sum(equals))

        accuracy += (torch.sum(equals).cpu().numpy())
    # print(equals)
    # print(target)

    # calculate average losses
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)
    accuracy = accuracy / len(valid_loader.dataset)

    train_loss_hist.append(train_loss)
    valid_loss_hist.append(valid_loss)

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

plt.plot(range(n_epochs), train_loss_hist, label="Training")
plt.plot(range(n_epochs), valid_loss_hist, label="Validation")
plt.legend()
plt.savefig('Loss with normalize.png')


# transforms2 = transforms.Compose([transforms.Resize(128),
#                                  transforms.RandomHorizontalFlip(),
#                                  transforms.ToTensor()
#                                 #  transforms.RandomResizedCrop(128),
                                 
#                                 #  transforms.Grayscale(num_output_channels=3),
#                                 #  transforms.Normalize((0,0,0), (1,1,1))
#                                 ])

test_data = Dataset.ImageFolder(data_dir_test, transform=transforms)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, drop_last=False)

f_out = open("submission.csv", "w+")
f_out2 = open("submission2.csv", "w+")

######################
# validate the model #
######################
model.eval()

index = 0
f_out.write("id,label\n")  
f_out2.write("id,label\n") 

for data, target in test_loader:

    # forward pass: compute predicted outputs by passing inputs to the model
    # print("type data : ", type(data[0]))
    # torch_data = torch.tensor(data)

    # if train_on_gpu:
    data = data.cuda()
    
    # output = model(torch.FloatTensor(torch.stack(data)))
    output = model(data)
    t = Variable(torch.FloatTensor([0.5]))  # threshold
    out = (output > t.cuda(async=True)).float() * 1
    out_t = out.t()
    

    # print("output ", out_t)
    for idx in range(len(out_t[0])):
        index +=1

        if out_t[0][idx] == 0.:
            f_out.write(str(index) + ",Cat\n")  
            f_out2.write(str(index) + ",Dog\n")
        else:
            f_out.write(str(index) + ",Dog\n")  
            f_out2.write(str(index) + ",Cat\n")

f_out.close()
f_out2.close()
