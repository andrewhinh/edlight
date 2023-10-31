import torch


def collator(batch, processor, max_length):
    new_batch = {"flattened_patches": [], "attention_mask": []}
    texts = [item["description"] for item in batch]
    text_inputs = processor(
        text=texts,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=True,
        max_length=max_length,
    )

    new_batch["labels"] = text_inputs.input_ids
    for item in batch:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])
    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

    return new_batch
