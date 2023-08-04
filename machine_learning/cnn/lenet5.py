# https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/
# 이 파일은 해당 블로그 게시글의 코드를 수정한 것입니다.
# mode = "print_forward"로 실행한 결과입니다 : 
#                                  Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))  →  torch.Size([64, 6, 28, 28])
#   BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  →  torch.Size([64, 6, 28, 28])
#                                                                           ReLU()  →  torch.Size([64, 6, 28, 28])
#       MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  →  torch.Size([64, 6, 14, 14])
#                                 Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))  →  torch.Size([64, 16, 10, 10])
#  BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  →  torch.Size([64, 16, 10, 10])
#                                                                           ReLU()  →  torch.Size([64, 16, 10, 10])
#       MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  →  torch.Size([64, 16, 5, 5])
#                                                                         Unwrap()  →  torch.Size([64, 400])
#                             Linear(in_features=400, out_features=120, bias=True)  →  torch.Size([64, 120])
#                                                                           ReLU()  →  torch.Size([64, 120])
#                              Linear(in_features=120, out_features=84, bias=True)  →  torch.Size([64, 84])
#                                                                           ReLU()  →  torch.Size([64, 84])
#                               Linear(in_features=84, out_features=10, bias=True)  →  torch.Size([64, 10])
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

batch_size = 64
num_classes = 10
learning_rate = 0.001 
num_epochs = 10
mode = "print_forward"
# mode = "train_test"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform = transforms.Compose([
                                                 transforms.Resize((32,32)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean = (0.1307,), std=(0.3081,))
                                           ]),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',
                                          transform = transforms.Compose([
                                              transforms.Resize((32,32)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean = (0.1325), std=(0.3105,))
                                          ]),
                                          download=True)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle=True)
class Unwrap(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.reshape(x.size(0), -1)

model = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
    nn.BatchNorm2d(6),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    Unwrap(),
    nn.Linear(400, 120),
    nn.ReLU(),
    nn.Linear(120, 84),
    nn.ReLU(),
    nn.Linear(84, num_classes)
).to(device)

if mode == 'print_forward':
    # pring forward results
    with torch.no_grad():
        handles = {}
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                handles[name] = module.register_forward_hook(lambda module, _, output: print(str(module).partition('\n')[0].rjust(90), " → ", output.shape))
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            break
elif mode == "test_train":
    cost = nn.CrossEntropyLoss() # Gaussian connection을 대체합니다.
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = cost(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 400 == 0:
                print( 'Epoch [{}/{}], Step [{}/{}], Loss : {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
