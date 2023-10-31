import evaluate
import torch
from tqdm.auto import tqdm


def run_evaluation(dataloader, model, processor, device, metric_name, max_length):
    metric = evaluate.load(metric_name)
    progress_bar = tqdm(range(len(dataloader)))

    model.to(device)
    model.eval()

    for batch in dataloader:
        labels = batch.pop("labels").to(device)
        flattened_patches = batch.pop("flattened_patches").to(device)
        attention_mask = batch.pop("attention_mask").to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                flattened_patches=flattened_patches,
                attention_mask=attention_mask,
                labels=labels,
                max_new_tokens=max_length,
            )

        predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        references = processor.batch_decode(labels, skip_special_tokens=True)
        metric.add_batch(predictions=predictions, references=references)
        progress_bar.update(1)

    results = metric.compute()
    return results
