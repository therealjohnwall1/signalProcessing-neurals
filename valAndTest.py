import torch
import torch.nn.functional as F
from train import load_batches, BATCH_SIZE, device
from models import digitRecog

MODEL_PATH = "numClassifier.pth"

def val_model(model, loader):
    # model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x,y in loader:
            X,Y = x.to(device), y.to(device)
            # not goot pratice ik
            if X.shape[0] < BATCH_SIZE:
                missing = BATCH_SIZE - X.shape[0]
                X = torch.cat((prev_X[-missing:], X), dim=0)
                Y = torch.cat((prev_Y[-missing:], Y), dim=0)
            prev_X, prev_Y = X, Y
            X = X.view(BATCH_SIZE,1,40,99)

            y_hat = model(X)
            loss = F.cross_entropy(y_hat, Y)
            total_loss += loss.item()
            _, predicted = torch.max(y_hat, 1)
            correct += (predicted == Y).sum().item()
            total += Y.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy



# batch size can be arb for this
if __name__ == "__main__":
    val_loader = load_batches(BATCH_SIZE, "dataset/forModel/val") 
    test_loader = load_batches(BATCH_SIZE, "dataset/forModel/test") 

    model = digitRecog().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Evaluating on Validation Data:")
    val_loss, val_accuracy = val_model(model, val_loader)
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    print("Evaluating on Test Data:")
    test_loss, test_accuracy = val_model(model, test_loader)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


   


    