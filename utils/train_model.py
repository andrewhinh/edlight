import torch
from tqdm.auto import tqdm
from transformers import get_scheduler


def run_training(
    dataloader,
    model,
    epochs,
    lr,
    device,
    model_save_path,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    progress_bar = tqdm(range(num_training_steps))

    model.to(device)
    model.train()

    for _ in range(epochs):
        for batch in dataloader:
            labels = batch.pop("labels").to(device)
            flattened_patches = batch.pop("flattened_patches").to(device)
            attention_mask = batch.pop("attention_mask").to(device)

            outputs = model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    # Save the model
    model.save_pretrained(model_save_path)
