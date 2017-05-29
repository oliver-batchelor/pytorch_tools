import torch.utils.data as data
import os
import os.path


imageExtensions = ['jpg', 'jpeg', 'png', 'ppm', 'bmp']

def has_extension(extensions, filename):
    return any(filename.lower().endswith("." + extension) for extension in extensions)


def image_with_mask(extensions):
    def f(filename):
        mask_filename = filename + ".mask"
        if(has_extension(extensions, filename) and os.path.exists(mask_filename)):
            return (filename, mask_filename)
    return f


def file_with_annotations(extensions, annotations):
    def f(filename):
        files = {a : mask_filename + "." + a for a in annotations}
        if(has_extension(extensions, filename) and all(map(os.path.exists), files.values())):
            return (filename, files)
    return f

def find_files(dir, file_filter):
    images = []
    for fname in os.listdir(dir):
        item = file_filter(os.path.join(dir, fname))
        if(item):
            images.append(item)

    return images


class FlatFolder(data.Dataset):

    def __init__(self, root, loader, file_filter = image_with_mask(imageExtensions), transform=None):

        self.root = root
        self.transform = transform
        self.loader = loader
        self.file_filter = file_filter

        self.rescan()


    def rescan(self):
        self.imgs = find_files(self.root, self.file_filter)
        if len(self.imgs) == 0:
            raise(RuntimeError("Found 0 matching images in: " + self.root + "\n"))



    def __getitem__(self, index):
        img, target = self.loader(*self.imgs[index])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
