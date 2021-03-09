import torch
from torch.utils.data import TensorDataset, DataLoader


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear_1 = torch.nn.Linear(D_in, H)
        self.linear_2 = torch.nn.Linear(H, D_out)

    def forward(self, data):
        h = self.linear_1(data)
        h_relu = torch.nn.functional.relu(h)
        y_pred = self.linear_2(h_relu)
        return y_pred


if __name__ == '__main__':
    device = torch.device('cpu')
    learning_rate = 1e-2

    x = torch.randn(64, 1000, device=device)
    y = torch.randn(64, 10, device=device)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    model = TwoLayerNet(D_in=1000, H=100, D_out=10)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epochs in range(50):
        for x_batch, y_batch in loader:
            y_prediction = model(x_batch)
            loss = torch.nn.functional.mse_loss(y_prediction, y_batch)

            print(loss.item())

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
