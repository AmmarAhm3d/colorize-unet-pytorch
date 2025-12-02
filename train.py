# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data import get_dataloader, lab_to_rgb_image
from model import UNet
from torchvision.utils import save_image
import os

# --- Hyperparameters ---
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "coco/images/val2017" # e.g., "coco_val2017"
SAVE_DIR = "saved_images"
CHECKPOINT_DIR = "checkpoints"

def train_epoch(model, dataloader, optimizer, loss_fn, epoch):
    loop = tqdm(dataloader, leave=True)
    total_loss = 0.0

    for i, batch in enumerate(loop):
        l_channel, ab_channels = batch['L'].to(DEVICE), batch['ab'].to(DEVICE)

        # Forward pass
        predicted_ab = model(l_channel)
        loss = loss_fn(predicted_ab, ab_channels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item(), epoch=epoch)

    return total_loss / len(dataloader)


def save_some_examples(model, val_loader, epoch, folder):
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        l_channel, ab_channels = batch['L'].to(DEVICE), batch['ab'].to(DEVICE)
        
        predicted_ab = model(l_channel)
        
        # Save original and predicted images
        for i in range(min(4, l_channel.size(0))): # Save up to 4 images
            original_rgb = lab_to_rgb_image(l_channel[i], ab_channels[i])
            predicted_rgb = lab_to_rgb_image(l_channel[i], predicted_ab[i])
            
            # Create a comparison image
            comparison = torch.cat([
                torch.from_numpy(original_rgb).permute(2,0,1), 
                torch.from_numpy(predicted_rgb).permute(2,0,1)
            ], dim=2)
            
            save_image(comparison.float()/255., os.path.join(folder, f"epoch_{epoch}_sample_{i}.png"))
    model.train()


def main():
    print(f"Using device: {DEVICE}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model = UNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.L1Loss() # L1 loss is good for colorization

    train_loader = get_dataloader(DATA_DIR, batch_size=BATCH_SIZE)
    # Use the same loader for validation for simplicity, or create a separate val set
    val_loader = get_dataloader(DATA_DIR, batch_size=4, shuffle=False)

    for epoch in range(NUM_EPOCHS):
        avg_loss = train_epoch(model, train_loader, optimizer, loss_fn, epoch)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {avg_loss:.4f}")

        # Save some visual results
        save_some_examples(model, val_loader, epoch, SAVE_DIR)
        
        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pth"))
    
    # Save final model
    torch.save(model.state_dict(), "final_colorization_model.pth")
    print("Training finished and final model saved.")

if __name__ == "__main__":
    main()