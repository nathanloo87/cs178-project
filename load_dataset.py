from imports import * 

# define the transformers
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# this one gonna load the dataset
train_set = SVHN(root='./data', split='train', download=True, transform=transform)
test_set = SVHN(root='./data', split='test', download=True, transform=transform)

# these are the data loaders
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# this one will check the image sampless
images, labels = next(iter(train_loader))
plt.imshow(images[0].permute(1, 2, 0))
plt.title(f"Label: {labels[0]}")
plt.show()

