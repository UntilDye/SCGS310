import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes, input_channels=1, hidden_size=256, num_layers=2):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.linear_before_rnn = nn.Linear(512, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        if h > 1:
            x = x.mean(2)
        else:
            x = x.squeeze(2)
        x = x.permute(0, 2, 1)  # (batch, width, channels)
        x = self.linear_before_rnn(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    import torch

    images = torch.randn(16, 1, 32, 100)  # (batch_size, channels, height, width)
    model = CRNN(num_classes=37, input_channels=1)  # 指定1通道
    outputs = model(images)
    print(f"Output shape: {outputs.shape}") # (16, width, num_classes)#