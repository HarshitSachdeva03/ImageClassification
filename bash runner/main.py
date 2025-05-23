import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model import CustomCNN
from utils import train_model, evaluate_model, get_data_loaders

def main(args):
    train_loader, val_loader, test_loader = get_data_loaders(args.data_path, args.img_size, args.batch_size)

    model = CustomCNN(args.conv_layers, args.fc_layers, args.num_classes, args.use_maxpool, args.stride).to("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    test_loss, test_acc = evaluate_model(model, test_loader, criterion)
    print(f"\n✅ Final Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Image/4")
    parser.add_argument("--conv_layers", nargs='+', type=int, default=[64, 128, 256, 512, 512])
    parser.add_argument("--fc_layers", nargs='+', type=int, default=[1024, 512])
    parser.add_argument("--stride", nargs='+', type=int, default=[2, 1, 1, 1, 1])
    parser.add_argument("--use_maxpool", action='store_true')
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--img_size", type=int, default=84)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=33)

    args = parser.parse_args()
    main(args)
