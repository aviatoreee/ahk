from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def get_train_data_loader(train_data_path, batch_size=4):
    train_data = ImageFolder(root=train_data_path, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    return train_loader
    
def get_test_data_loader(test_data_path=None, batch_size=4):

    test_data = ImageFolder(root=test_data_path, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return test_loader
