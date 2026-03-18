import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from utils import ImprovedMNISTFeatureExtractor

# Configuration
DEVICE = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 10  # More epochs for better learning
FEATURE_DIM = 512  # Feature dimension (comparable to VGG)
SAVE_PATH = 'checkpoints/cnn_mnist_features_extractor.pkl'


def evaluate(model, test_loader, device):
    """Evaluate accuracy on the test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, return_features=False)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100 * correct / total


def main():
    os.makedirs('checkpoints', exist_ok=True)

    print(f"🚀 Training Feature Extractor on {DEVICE}")
    print(f"📊 Feature dimension: {FEATURE_DIM}")

    # Data augmentation to improve feature quality
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, 
                                   transform=transform_train)
    test_dataset = datasets.MNIST('./data', train=False, 
                                  transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=2)
    
    # Improved model
    model = ImprovedMNISTFeatureExtractor(feature_dim=FEATURE_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data, return_features=False)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        # Evaluation on the test set
        test_acc = evaluate(model, test_loader, DEVICE)
        train_acc = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")
        
        # Save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'feature_dim': FEATURE_DIM
            }, SAVE_PATH)
            print(f"✅ Best model saved (Test Acc: {test_acc:.2f}%)")
        
        scheduler.step()
    
    print(f"\n🎯 Training completed! Best accuracy: {best_acc:.2f}%")
    print(f"💾 Model saved at: {SAVE_PATH}")


if __name__ == "__main__":
    main()