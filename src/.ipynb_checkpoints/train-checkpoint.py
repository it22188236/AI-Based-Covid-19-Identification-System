import torch
from torch.utils.data import DataLoader
from dataset import COVIDAudioDataset
from model import SimpleCNN
from augmentation import get_augmentation_pipeline
from utils import load_config, set_seed
import os

def main():
    config = load_config("../configs/config.yaml")
    set_seed()

    noises_dir = os.path.join(config['data']['root'], config['data']['noises_dir'])
    augment_pipeline = get_augmentation_pipeline(noises_dir, config)

    train_dataset = COVIDAudioDataset(
        csv_file=f"../data/splits/train.csv",
        config=config,
        augment_pipeline=augment_pipeline,
        is_train=True
    )
    val_dataset = COVIDAudioDataset(
        csv_file=f"../data/splits/val.csv",
        config=config,
        augment_pipeline=None,
        is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    best_auc = 0
    for epoch in range(config['training']['num_epochs']):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Simple validation
        model.eval()
        # Add your validation loop here if needed

        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} completed.")

    torch.save(model.state_dict(), config['training']['model_save_path'])
    print("Training finished. Model saved.")

if __name__ == "__main__":
    main()