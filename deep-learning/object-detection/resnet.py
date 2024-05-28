from torchvision.transforms import ToPILImage
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.manifold import TSNE
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 30
batch_size = 100
learning_rate = 0.001
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

train_dataset = torchvision.datasets.CIFAR10(
    root='Data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(
    root='Data', train=False, transform=transforms.ToTensor(), download=False)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)

print('train size:{}, test size:{}'.format(
    len(train_loader), len(test_loader)))

show = ToPILImage()  # 可以把Tensor转成Image，方便可视化
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0.2, wspace=0.1)
for i in range(6):
    (image, label) = test_dataset[i]
    ax = fig.add_subplot(2, 3, i+1, xticks=[], yticks=[])
    plt.title('{}'.format(classes[label]))
    ax.imshow(show(image))


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channel = 16
        self.conv1 = nn.Conv2d(3, 16, stride=1, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.block1 = self.make_layer(block, 16, 1)
        self.block2 = self.make_layer(block, 16, 1)
        self.block3 = self.make_layer(block, 32, 2)
        self.block4 = self.make_layer(block, 32, 1)
        self.block5 = self.make_layer(block, 64, 2)
        self.block6 = self.make_layer(block, 64, 1)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channel, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channel != out_channel):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel,
                          kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channel))
        out_layer = block(self.in_channel, out_channel, stride, downsample)
        self.in_channel = out_channel
        return out_layer

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


resnet = ResNet(ResidualBlock).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=learning_rate)


def update_lr(optimizer, lr):
    for para in optimizer.param_groups:
        para['lr'] = lr


total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # print(images.shape)
        outputs = resnet(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (idx+1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch +
                  1, num_epochs, idx+1, total_step, loss.item()))

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = resnet(images)
        predicted = torch.argmax(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(
        100 * correct / total))

# visualization of trained flatten layer (t-SNE)
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 500
low_dim_embs = tsne.fit_transform(outputs.data.cpu().numpy())[:plot_only, :]
plot_labels = labels.cpu().numpy()[:plot_only]
# plot_with_labels(low_dim_embs, plot_labels)


def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, classes[s], backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.show()
    plt.pause(0.01)
