from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10



class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None: 
            pos_1 = self.transform[0](img)
            pos_2 = self.transform[1](img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target
