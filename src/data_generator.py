import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
from image_helper import crop_image, rescale_intensity

class DataGenerator(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=4, image_size=192):
        self.ids = ids # list of tuples (id, frame)
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()

    def __load__(self, id_name, fr):
        ## Path
        data_dir = os.path.join(self.path, id_name)

        image_name = '{0}/{1}_sa_{2}.nii.gz'.format(data_dir, id_name, fr)
        label_name = '{0}/{1}_sa_gt_{2}.nii.gz'.format(data_dir, id_name, fr)

        ## Reading image
        nim = nib.load(image_name)
        image = nim.get_fdata()
        nil = nib.load(label_name)
        label = nil.get_fdata()

        X, Y, _ = image.shape
        cx, cy = int(X / 2), int(Y / 2)
        image = crop_image(image, cx, cy, self.image_size)
        label = crop_image(label, cx, cy, self.image_size)

        ## Intensity rescaling
        image = rescale_intensity(image, (1.0, 99.0))

        return image, label

    def __getitem__(self, index):
        if (index+1) * self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index * self.batch_size

        files_batch = self.ids[index * self.batch_size : (index+1) * self.batch_size]

        images = []
        masks  = []

        for id_name, frame in files_batch:
            image, mask = self.__load__(id_name, frame)

            _, _, Z = image.shape

            # Append the image slices to the batch
            # Use list for appending, which is much faster than numpy array
            for z in range(Z):
                images += [image[:, :, z]]
                masks += [mask[:, :, z]]

        images = np.array(images)
        masks  = np.array(masks)

        return image, mask

    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))
