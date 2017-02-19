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