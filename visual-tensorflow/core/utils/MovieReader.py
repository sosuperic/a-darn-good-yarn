# Read frames from a movie and write to file

import argparse
import cv2
import numpy as np
import os

class MovieReader(object):
    def __init__(self):
        pass

    def get_fps(self, vidcap):
        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
         
        if int(major_ver)  < 3 :
            fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
        else :
            fps = vidcap.get(cv2.CAP_PROP_FPS)
        print 'Frames per second: {0}'.format(fps)
        return fps

    def get_num_frames(self, vidcap):
        length = int(vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        print 'Number of frames {}'.format(length)
        return length

    def get_frame_indices(self, fps, num_frames, sample_rate_in_sec):
        """Return list of indices corresponding to a frame every sample_rate_in_sec"""
        indices = []
        cur_idx_approx = 0.0
        while cur_idx_approx < num_frames:
            cur_idx_approx += fps * sample_rate_in_sec
            indices.append(int(round(cur_idx_approx)))
        return indices

    def get_timestamp_str(self, frame_idx, fps):
        """Return '1h4m36s'"""
        seconds = float(frame_idx) / fps
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        timestamp_str = '{}h{:02}m{:02}s'.format(int(h),int(m),int(s))
        return timestamp_str

    def resize_and_center_crop(self, frame, target_w, target_h):
        """Resize while keeping aspect ratio so that (w >= target_w and h >= target_h).
        Then center crop. Frame may be smaller or larger than targets.
        """
        h, w, n_channels = frame.shape

        # Resizing
        # This if-else works for both resizing to make image smaller or larger.
        # - If (at least) one dim is smaller than the target, then the *relatively* smaller dim
        # will be resized to the target size, with the aspect ratio ensuring that the *relatively*
        # larger dim will be greater than its target size as well.
        # - If both larger than the targets, then the *relatively* not as large dim will 
        # be resized to the target size, maximizing what's 'left over'. We want as much 
        # left over as possible before the central crop.
        h2w_ratio = float(h) / w
        h2target_h, w2target_w = h / target_h, w / target_w
        if h2target_h < w2target_w:
            new_h = target_h
            new_w = int(target_h / h2w_ratio)
        else:
            new_w = target_w
            new_h = int(target_w * h2w_ratio)
        frame = cv2.resize(frame, (new_w, new_h))

        # result = frame
        # Central crop
        if new_h == target_h:                       # width must be cropped or filled
            w_offset = int(round((abs(target_w - new_w)) / 2.0))
            if new_w < target_w:                    # fill sides
                result = np.zeros([target_h, target_w, n_channels])
                result[:,w_offset:w_offset+target_w,:] = frame
            else:                                   # crop off sides
                result = frame[:,w_offset:w_offset+target_w,:]
        else:                                       # crop off top and bottom
            h_offset = int(round((target_h - new_h) / 2.0))
            if new_h < target_h:                    # fill sides
                result = np.zeros([target_h, target_w, n_channels])
                result[h_offset:h_offset+target_h,:,:] = frame
            else:
                result = frame[h_offset:h_offset+target_h,:,:]
        return result

    def write_frames(self, input_path, output_dir, sample_rate_in_sec, target_w, target_h):
        vidcap = cv2.VideoCapture(input_path)
        fps = self.get_fps(vidcap)
        num_frames = self.get_num_frames(vidcap)
        frame_indices = self.get_frame_indices(fps, num_frames, sample_rate_in_sec)


        # Make output directory if not exists
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(input_path), 'frames')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        # Write frames to dir
        for i, frame_idx in enumerate(frame_indices):
            vidcap.set(1, frame_idx)
            success, frame = vidcap.read()
            if success:
                frame = self.resize_and_center_crop(frame, target_w, target_h)
                timestamp_str = self.get_timestamp_str(frame_idx, fps)
            #         image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)

                # We use the i in the file name so that it sorts in order lexicographically
                # Also the timestamp str rounds the seconds, so to prevent overwriting
                out_fn = os.path.join(output_dir, 'frame_{}_{}.jpg'.format(i, timestamp_str))

                cv2.imwrite(out_fn, frame)

        print 'Saving a frame every {} second(s)'.format(sample_rate_in_sec)
        print 'Total number of frames saved: {}'.format(len(frame_indices))

        vidcap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read a movie, write frames, etc.')

    # Defaults
    parser.add_argument('-i', '--input_path', dest='input_path', default=None)
    parser.add_argument('-o', '--output_dir', dest='output_dir', default=None)
    parser.add_argument('-sr', '--sample_rate_in_sec', dest='sample_rate_in_sec', default=1)
    parser.add_argument('-tw', '--target_w', dest='target_w', default=256)
    parser.add_argument('-th', '--target_h', dest='target_h', default=256)
    args = parser.parse_args()
 
    mr = MovieReader()
    mr.write_frames(args.input_path, args.output_dir, int(args.sample_rate_in_sec), args.target_w, args.target_h)
