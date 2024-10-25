import torch
import wandb
import os
import gc
from torch import optim
from torch import nn
from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm
from safetensors.torch import save_model
from .utils_config import config, count_params
from .model import mobilenet
from .dataload import train_loader

criterion = nn.CrossEntropyLoss()  # loss function
optimizer = optim.AdamW(params=mobilenet.parameters(), lr=config.lr)
scaler = GradScaler()
epochs = config.epoch_count


param_count = count_params(mobilenet)
print(param_count)


# initilaize wandb
wandb.login()
train_run = wandb.init(project="HAR", name="har_mobilenet_1")
wandb.watch(mobilenet)

torch.cuda.empty_cache()
gc.collect()


def training_loop(
    model=mobilenet, train_loader=train_loader, epochs=epochs, config=config
):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    accuracy = 0.0

    for epoch in tqdm(range(epochs + 1)):
        optimizer.zero_grad()

        torch.cuda.empty_cache()
        print(f"Training epoch {epoch}")

        for x, (image, label) in tqdm(enumerate(train_loader)):
            image = image.to(config.device)
            label = label.to(config.device)

            # every iterations
            torch.cuda.empty_cache()
            gc.collect()

            # Mixed precision training
            with torch.autocast(device_type="cuda", dtype=config.dtype):
                output = model(image)
                train_loss = criterion(output, label.long())
                train_loss = train_loss / config.grad_acc_step  # Normalize the loss

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            accuracy = 100 * correct / total

            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(train_loss).backward()

            if (x + 1) % config.grad_acc_step == 0:
                # Unscales the gradients of optimizer's assigned params in-place

                scaler.step(optimizer)
                # Updates the scale for next iteration
                scaler.update()
                optimizer.zero_grad()

            if (x + 1) % 5 == 0:
                wandb.log({"loss": train_loss, "accuracy": accuracy})

        print(
            f"Epoch {epoch} of {epochs}, train_loss: {train_loss.item():.4f}, accuracy: {accuracy:.4f}"
        )

        print(f"Epoch @ {epoch} complete!")

    print(
        f"End metrics for run of {epochs}, train_loss: {train_loss.item():.4f}, accuracy: {accuracy:.4f}"
    )

    save_model(model, config.safetensor_file)


training_loop()
torch.cuda.empty_cache()
gc.collect()


print("mobilenet for Human Action Recognition pretraining complete")
# Ciao
