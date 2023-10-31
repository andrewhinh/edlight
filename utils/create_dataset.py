from PIL import Image
from torch.utils.data import Dataset


class ImageCaptioningDataset(Dataset):
    """Dataset for image captioning."""

    def __init__(self, dataset, max_img_size, processor, max_patches):
        self.dataset = dataset
        self.max_img_size = max_img_size
        self.processor = processor
        self.max_patches = max_patches

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset.iloc[idx]
        image = Image.open(item["file"]).resize((self.max_img_size, self.max_img_size))
        encoding = self.processor(
            images=image, return_tensors="pt", add_special_tokens=True, max_patches=self.max_patches
        )
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["description"] = item["description"]
        return encoding
