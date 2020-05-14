import torchvision.transforms as transforms

# data augmentation for training and test time

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Resize, normalize and jitter image brightness
data_jitter_brightness = transforms.Compose([
    transforms.ColorJitter(brightness=0),
    transforms.ColorJitter(brightness=5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Resize, normalize and jitter image saturation
data_jitter_saturation = transforms.Compose([
    transforms.ColorJitter(saturation=5),
    transforms.ColorJitter(saturation=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Resize, normalize and jitter image contrast
data_jitter_contrast = transforms.Compose([
    transforms.ColorJitter(contrast=5),
    transforms.ColorJitter(contrast=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Resize, normalize and jitter image hues
data_jitter_hue = transforms.Compose([
    transforms.ColorJitter(hue=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Resize, normalize and rotate image
data_rotate = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Resize, normalize and flip image horizontally and vertically
data_hvflip = transforms.Compose([
    transforms.RandomHorizontalFlip(1),
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Resize, normalize and flip image horizontally
data_hflip = transforms.Compose([
    transforms.RandomHorizontalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Resize, normalize and flip image vertically
data_vflip = transforms.Compose([
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Resize, normalize and shear image
data_shear = transforms.Compose([
    transforms.RandomAffine(degrees = 15,shear=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



# Resize, normalize and convert image to grayscale
data_grayscale = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
