# Prepare the data (parse, download, utilities, etc.)

import cv2

########################################################################################################################
# Utility functions
########################################################################################################################

def get_grayscale_hist(img, bins=64):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    return hist

def get_color_hist(img, bins=64):
    chans = cv2.split(img)
    colors = ("b", "g", "r")

    features = []
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and
        # concatenate the resulting histograms for each
        # channel
        hist = cv2.calcHist([chan], [0], None, [bins], [0, 256])
        features.extend(hist)

    return features


if __name__ == '__main__':
    import numpy as np
    img_batch = np.random.randint(256, size=[2,256,256,3], dtype=np.uint8)
    get_grayscale_hist(img_batch[0])

    img_batch_feats = np.zeros([2, 64])
    for i in range(img_batch.shape[0]):
        img_batch_feats[i] = np.squeeze(get_grayscale_hist(img_batch[i]))

    print img_batch_feats