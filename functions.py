import torch
from tqdm.auto import tqdm
from helper_functions import accuracy_fn
def train_step(model, dataloader, optimizer, loss_fn, accuracy_fn, device):

    train_loss, train_acc = 0, 0
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        acc = accuracy_fn(y_true= y,
                          y_pred= y_pred.argmax(dim= 1))
        train_acc += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    print(f"Train loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}%| ")

def test_step(model, dataloader, optimizer, loss_fn, accuracy_fn, device):

    test_loss, test_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            test_pred = model(X)

            loss  = loss_fn(test_pred, y)
            test_loss += loss.item()

            acc = accuracy_fn(y_true= y,
                              y_pred= test_pred.argmax(dim= 1))
            test_acc += acc

        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
    print(f"Test loss: {test_loss:.2f} | Test Acc: {test_acc:.2f}%\n")

def eval_mode(model, dataloader, optimizer, loss_fn, accuracy_fn, device):
    loss, acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(dataloader):

            y = torch.tensor(y)

            y_pred = model(X)

            loss += loss_fn(y_pred, y)

            acc += accuracy_fn(y_true= y,
                               y_pred= y_pred.argmax(dim= 1))

        loss /= len(dataloader)
        acc /= len(dataloader)

    return{"model_name": model.__class__.__name__,
         "model_loss": loss.item(),
         "model_acc": acc}

def make_predictions(model, data, device):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim= 0).to(device)

            pred_logits = model(sample)

            pred_prob = torch.softmax(pred_logits, dim= 1)

            pred_probs.append(pred_prob.cpu())

    return torch.cat(pred_probs)

