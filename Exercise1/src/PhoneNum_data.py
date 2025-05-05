import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

class PhoneNumberDataset(Dataset):
    def __init__(self, num_samples=20000, is_balanced=True, transform=None, save_dir='./Dataset/'):
        self.num_samples = num_samples
        self.is_balanced = is_balanced
        self.transform = transform
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.country_codes = ['01', '44', '91', '86', '81', '49', '33', '61', '55', '82']
        self.data = self._generate_data()
    
    def _generate_phone_number(self, is_balanced=True):
        if is_balanced:
            while True:
                phone_number = random.choices('0123456789', k=10)
                digit_counts = {d: phone_number.count(d) for d in set(phone_number)}
                if all(count <= 4 for count in digit_counts.values()):
                    return ''.join(phone_number)
        else:
            repeated_digits = random.choices('0123456789', k=4)
            other_digits = random.choices('0123456789', k=6)
            positions = random.sample(range(10), 4)
            phone_number = [''] * 10
            digit_idx = 0
            for pos in positions:
                phone_number[pos] = repeated_digits[digit_idx]
                digit_idx += 1
            other_idx = 0
            for i in range(10):
                if phone_number[i] == '':
                    phone_number[i] = other_digits[other_idx]
                    other_idx += 1
            return ''.join(phone_number)
    
    def _create_image(self, phone_number, add_country_code=False):
        image_width = 280
        image_height = 28
        image = Image.new('L', (image_width, image_height), color=255)
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        if add_country_code:
            country_code = random.choice(self.country_codes)
            phone_number = country_code + phone_number
        for i, digit in enumerate(phone_number):
            x_position = i * (image_width // len(phone_number)) + 5
            draw.text((x_position, 2), digit, font=font, fill=0)
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_tensor = torch.tensor(img_array).unsqueeze(0)
        return img_tensor
    
    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            phone_number = self._generate_phone_number(self.is_balanced)
            add_country_code = random.random() > 0.5
            image = self._create_image(phone_number, add_country_code)
            data.append({
                'phone_number': phone_number,
                'image': image,
                'label': torch.tensor([int(digit) for digit in phone_number], dtype=torch.long)
            })
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def save_dataset(self, filename):
        torch.save(self.data, os.path.join(self.save_dir, filename))
        print(f"Dataset saved to {os.path.join(self.save_dir, filename)}")
    
    @classmethod
    def load_dataset(cls, filepath, transform=None):
        data = torch.load(filepath)
        dataset = cls(num_samples=1, transform=transform)
        dataset.data = data
        dataset.num_samples = len(data)
        return dataset

    def visualize_samples(self, num_samples=5):
        indices = random.sample(range(len(self.data)), min(num_samples, len(self.data)))
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        for i, idx in enumerate(indices):
            img = self.data[idx]['image'].squeeze(0).numpy()
            label = ''.join([str(l.item()) for l in self.data[idx]['label']])
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Phone: {label}")
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

def get_data_loaders(balanced_path=None, imbalanced_path=None, batch_size=64, train_ratio=0.7, val_ratio=0.15, 
                     create_new=True, num_samples=20000, save_dir='./Dataset/'):
    transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))
    ])
    if create_new:
        balanced_dataset = PhoneNumberDataset(num_samples=num_samples, is_balanced=True, transform=transform, save_dir=save_dir)
        balanced_filename = "phone_balanced.pt"
        balanced_dataset.save_dataset(balanced_filename)
        balanced_path = os.path.join(save_dir, balanced_filename)
        imbalanced_dataset = PhoneNumberDataset(num_samples=num_samples, is_balanced=False, transform=transform, save_dir=save_dir)
        imbalanced_filename = "phone_imbalanced.pt"
        imbalanced_dataset.save_dataset(imbalanced_filename)
        imbalanced_path = os.path.join(save_dir, imbalanced_filename)
    else:
        balanced_dataset = PhoneNumberDataset.load_dataset(balanced_path, transform=transform)
        imbalanced_dataset = PhoneNumberDataset.load_dataset(imbalanced_path, transform=transform)
    def split_indices(dataset_size):
        indices = list(range(dataset_size))
        random.shuffle(indices)
        train_end = int(dataset_size * train_ratio)
        val_end = train_end + int(dataset_size * val_ratio)
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        return train_indices, val_indices, test_indices
    b_train_idx, b_val_idx, b_test_idx = split_indices(len(balanced_dataset))
    i_train_idx, i_val_idx, i_test_idx = split_indices(len(imbalanced_dataset))
    b_train_subset = Subset(balanced_dataset, b_train_idx)
    b_val_subset = Subset(balanced_dataset, b_val_idx)
    b_test_subset = Subset(balanced_dataset, b_test_idx)
    i_train_subset = Subset(imbalanced_dataset, i_train_idx)
    i_val_subset = Subset(imbalanced_dataset, i_val_idx)
    i_test_subset = Subset(imbalanced_dataset, i_test_idx)
    loaders = {
        'balanced': {
            'train': DataLoader(b_train_subset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g),
            'val': DataLoader(b_val_subset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g),
            'test': DataLoader(b_test_subset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        },
        'imbalanced': {
            'train': DataLoader(i_train_subset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g),
            'val': DataLoader(i_val_subset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g),
            'test': DataLoader(i_test_subset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        },
        'cross': {
            'balanced_train_imbalanced_test': DataLoader(i_test_subset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g),
            'imbalanced_train_balanced_test': DataLoader(b_test_subset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        }
    }
    return loaders, balanced_path, imbalanced_path

if __name__ == "__main__":
    save_dir = './Dataset/'
    os.makedirs(save_dir, exist_ok=True)
    balanced_dataset = PhoneNumberDataset(num_samples=1000, is_balanced=True, save_dir=save_dir)
    balanced_dataset.save_dataset("phone_balanced_sample.pt")
    balanced_dataset.visualize_samples(5)
    imbalanced_dataset = PhoneNumberDataset(num_samples=1000, is_balanced=False, save_dir=save_dir)
    imbalanced_dataset.save_dataset("phone_imbalanced_sample.pt")
    imbalanced_dataset.visualize_samples(5)
    loaders, balanced_path, imbalanced_path = get_data_loaders(create_new=True, num_samples=20000, batch_size=64, save_dir=save_dir)
    print(f"Balanced dataset path: {balanced_path}")
    print(f"Imbalanced dataset path: {imbalanced_path}")
    train_iter = iter(loaders['balanced']['train'])
    images, labels = next(train_iter)
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"First few labels: {labels[0]}")
