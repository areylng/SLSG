import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms

class Repeat(Dataset):
    def __init__(self, org_dataset, new_length):
        self.org_dataset = org_dataset
        self.org_length = len(self.org_dataset)
        self.new_length = new_length

    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        return self.org_dataset[idx % self.org_length]

class MVTecAT(Dataset):
    def __init__(self, root_dir, defect_name, size, transform=None, mode="train",model_name='s',all_train_image_names=None,all_train_imgs=None):
        self.root_dir = Path(root_dir)
        self.defect_name = defect_name
        self.transform = transform
        self.mode = mode
        self.size = size


        self.test_transform = transforms.Compose([])
        self.test_transform.transforms.append(transforms.ToTensor())
        
        # find test images
        if self.mode == "train":
            self.image_names = all_train_image_names
            self.imgs = all_train_imgs
        else:
            #test mode
            self.image_names = sorted((self.root_dir / defect_name / "test").glob(str(Path("*") / "*.png")))
            #self.imagemask_names = list((self.root_dir / defect_name / "ground_truth").glob(str(Path("*") / "*.png")))
            
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            img = self.imgs[idx].copy()
            if self.transform is not None:
                img = self.transform(img)
            return img
        else:
            filename = self.image_names[idx]
            try:
                temp = self.image_names[idx]
                temp = str(temp)
                list_path = temp.split('\\')
                list_path[5] = 'ground_truth'
                Path_mask=''
                for ii in list_path:
                    Path_mask+=ii
                    Path_mask += "\\"
                Path_mask = Path_mask[0:-5]
                Path_mask += '_mask.png'
                img_mask = Image.open(Path_mask).convert('L')
                img_mask = img_mask.resize((256,256))
                img_mask = self.test_transform(img_mask)
            except:
                img_mask = torch.zeros((1,256,256))

            label = filename.parts[-2]
            img = Image.open(filename).convert("RGB")
            #img = img.resize((self.size,self.size))
            if self.transform is not None:
                img = self.transform(img)
            return img, label != "good",img_mask
