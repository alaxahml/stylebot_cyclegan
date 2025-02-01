import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CycleDataset(Dataset):
    def __init__(self, root_B, root_A):
        self.root_B = root_B
        self.root_A = root_A
        self.transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.B_images = os.listdir(root_B)
        self.A_images = os.listdir(root_A)
        self.length_dataset = max(len(self.B_images), len(self.A_images))
        self.B_len = len(self.B_images)
        self.A_len = len(self.A_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        B_img = self.B_images[index % self.B_len]
        A_img = self.A_images[index % self.A_len]

        B_path = os.path.join(self.root_B, B_img)
        A_path = os.path.join(self.root_A, A_img)

        # B_img = np.array(Image.open(B_path).convert("RGB"))
        # A_img = np.array(Image.open(A_path).convert("RGB"))

        B_img = Image.open(B_path).convert("RGB")
        A_img = Image.open(A_path).convert("RGB")

        B_img = self.transform(B_img)
        A_img = self.transform(A_img)

        return B_img, A_img