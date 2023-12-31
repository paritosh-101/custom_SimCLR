from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from torchvision.datasets import ImageFolder
from torchvision.transforms import Grayscale


class ContrastiveLearningDataset:
    def __init__(self, root_folder, crop_size=32):
        self.root_folder = root_folder
        self.crop_size = crop_size

    # @staticmethod
    # def get_simclr_pipeline_transform(size, s=1):
    #     """Return a set of data augmentation transformations as described in the SimCLR paper."""
    #     color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    #     data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
    #                                           transforms.RandomHorizontalFlip(),
    #                                           transforms.RandomApply([color_jitter], p=0.8),
    #                                           transforms.RandomGrayscale(p=0.2),
    #                                           GaussianBlur(kernel_size=int(0.1 * size)),
    #                                           transforms.ToTensor()])
    #     return data_transforms
    
    # @staticmethod
    def get_simclr_pipeline_transform(self, s=1, is_grayscale=False):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        transformations = [transforms.RandomResizedCrop(size=self.crop_size),
                           transforms.RandomHorizontalFlip()]

        if not is_grayscale:
            # Color jitter only for RGB images
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            transformations.append(transforms.RandomApply([color_jitter], p=0.8))
            transformations.append(transforms.RandomGrayscale(p=0.2))

        # Gaussian blur can be applied to both grayscale and RGB
        transformations.append(GaussianBlur(kernel_size=int(0.1 * self.crop_size)))
        transformations.append(transforms.ToTensor())

        return transforms.Compose(transformations)

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(),
                                                              n_views),
                                                          download=True),
                                                          
                            # Custom RGB Dataset
                            'custom_rgb': lambda: ImageFolder(
                                self.root_folder,
                                transform=ContrastiveLearningViewGenerator(
                                    self.get_simclr_pipeline_transform(),  # Adjust size as needed
                                    n_views)
                            ),

                            # Custom Grayscale Dataset
                            'custom_grayscale': lambda: ImageFolder(
                                self.root_folder,
                                transform=transforms.Compose([
                                    Grayscale(num_output_channels=1),
                                    ContrastiveLearningViewGenerator(
                                        self.get_simclr_pipeline_transform(is_grayscale=True),  # Adjust size as needed
                                        n_views)
                                ])
                            )
                        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
