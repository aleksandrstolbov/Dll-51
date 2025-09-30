import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import copy
from PIL import Image

transform = transforms.Compose([transforms.ToTensor()])
data_train = datasets.MNIST('./data',
                            train = True,
                            download = False,
                            transform = transform
                           )
data_test = datasets.MNIST('./data',
                           train = False,
                           download = False,
                           transform = transform
                          )

traindata = DataLoader(data_train, batch_size = 100, shuffle = True)
testdata = DataLoader(data_test, batch_size = 100, shuffle = True)

dataiter = iter(traindata)
data = next(dataiter)
features, labels = data

print('Размер изображения', features[0].shape)
print('Количество классов', len(torch.unique(labels)))

# plt.imshow(torch.squeeze(features[0]), cmap = 'gray')
# plt.title('Класс '+ str(labels[0].item()))
# plt.show()

model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 32, kernel_size = 3, padding = 1, padding_mode = 'replicate'),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.ReLU(),
    torch.nn.Conv2d(32, 64, kernel_size = 3, padding = 1, padding_mode = 'replicate'),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.ReLU(),
    torch.nn.Conv2d(64, 128, kernel_size = 3, padding = 1, padding_mode = 'replicate'),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.ReLU(),
    torch.nn.Flatten(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(1152, 10)
    )
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print(f"Model is on device: {next(model.parameters()).device}")

# loss = torch.nn.CrossEntropyLoss()
# trainer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# num_epochs = 10
# for epoch in range(1, num_epochs + 1):
#     losses_per_epoch = []
#     for X, y in traindata:
#         X, y = X.to(device), y.to(device)
#         trainer.zero_grad()
#         l = loss(model(X), y)
#         losses_per_epoch.append(l.item())
#         l.backward()
#         trainer.step()
#     if epoch % 5 == 0:
#         print('epoch %d, loss: %f' % (epoch, np.mean(losses_per_epoch)))

# Загрузить модель
model.load_state_dict(torch.load('mnist_model.pth'))

y_pred = []
x_pred = []
# y_true = []
# for X, y in testdata:
#     X, y = X.to(device), y.to(device)
#     y_pred_iter = model(X)
#     y_pred.extend(y_pred_iter.detach())
#     y_true.extend(y.detach())
#
# y_pred = torch.Tensor([torch.argmax(x) for x in y_pred])
# y_true = torch.Tensor(y_true)
#
# from sklearn.metrics import accuracy_score
# print('Точность на тестовой выборке:', accuracy_score(y_true, y_pred))


# for index in range(10):
#     x_pred = features[index].unsqueeze(dim=1).clone().to(device)
#     y_pred = model(x_pred)
#
#     # y = torch.Tensor([torch.argmax(x) for x in y_pred])
#     y = torch.argmax(y_pred).item()
#
#     print('Класс '+ str(labels[index].item()) + '; Предсказание: ' + str(y))

# Загрузка изображения и преобразование в тензор
img = Image.open("9_1.jpg")
# Добавьте преобразование в Grayscale
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),  # Конвертация в 1 канал
    transforms.ToTensor()
])
tensor_img = transform(img)  # Теперь размер [1, 28, 28]

plt.imshow(torch.squeeze(tensor_img), cmap='gray')  # squeeze убирает размерность канала
plt.title('Класс '+ str(labels[0].item()))
plt.show()

x_pred = tensor_img.unsqueeze(dim=1).clone().to(device)
y_pred = model(x_pred)

# y = torch.Tensor([torch.argmax(x) for x in y_pred])
y = torch.argmax(y_pred).item()

print(img.filename + ';\nПредсказание: ' + str(y))
for i, yi in enumerate(torch.squeeze(y_pred)):
    print(f"Индекс {i}: {yi.item():.4f}")

# for index in range(10):
#     x_pred = features[index].unsqueeze(dim=1).clone().to(device)
#     y_pred = model(x_pred)
#
#     # y = torch.Tensor([torch.argmax(x) for x in y_pred])
#     y = torch.argmax(y_pred).item()
#
#     print('Класс '+ str(labels[index].item()) + '; Предсказание: ' + str(y))