import torch


x = torch.tensor(2)
y = torch.tensor(3)

w = torch.tensor(1.0, requires_grad=True)

y_hat = w * x
loss = (y_hat - y)**2
loss.backward()
print(w.grad)
