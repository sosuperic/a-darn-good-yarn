# Prepare the data (parse, download, utilities, etc.)

import cv2

from core.utils.utils import get_grayscale_hist, get_color_hist

########################################################################################################################
# Utility functions
########################################################################################################################

if __name__ == '__main__':
    import numpy as np
    img_batch = np.random.randint(256, size=[2,256,256,3], dtype=np.uint8)
    get_grayscale_hist(img_batch[0])

    img_batch_feats = np.zeros([2, 64])
    for i in range(img_batch.shape[0]):
        img_batch_feats[i] = np.squeeze(get_grayscale_hist(img_batch[i]))

    print img_batch_feats