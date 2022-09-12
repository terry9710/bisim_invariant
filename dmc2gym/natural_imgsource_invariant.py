
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import cv2
import skvideo.io
import random
import tqdm

class BackgroundMatting(object):
    """
    Produce a mask by masking the given color. This is a simple strategy
    but effective for many games.
    """
    def __init__(self, color):
        """
        Args:
            color: a (r, g, b) tuple or single value for grayscale
        """
        self._color = color

    def get_mask(self, img):
        return img == self._color


class ImageSource(object):
    """
    Source of natural images to be added to a simulated environment.
    """
    def get_image(self):
        """
        Returns:
            an RGB image of [h, w, 3] with a fixed shape.
        """
        pass

    def reset(self):
        """ Called when an episode ends. """
        pass


class FixedColorSource(ImageSource):
    def __init__(self, shape, color):
        """
        Args:
            shape: [h, w]
            color: a 3-tuple
        """
        self.arr = np.zeros((shape[0], shape[1], 3))
        self.arr[:, :] = color

    def get_image(self):
        return self.arr


class RandomColorSource(ImageSource):
    def __init__(self, shape, max_images=10):
        """
        Args:
            shape: [h, w]
        """
        self.color_list = np.random.randint(0, 256, size=(max_images, 3))
        print(self.color_list)
        self.color_id = None
        self.shape = shape
        self.arr = None
        self.reset()

    def reset(self):
        self.color_id = np.random.randint(0, len(self.color_list))
        self._color = self.color_list[self.color_id]
        self.arr = np.zeros((self.shape[0], self.shape[1], 3))
        self.arr[:, :] = self._color

    def get_image(self):
        return self.arr

    def get_envindex(self):
        env_index = self.color_id
        return env_index


class NoiseSource(ImageSource):
    def __init__(self, shape, strength=255):
        """
        Args:
            shape: [h, w]
            strength (int): the strength of noise, in range [0, 255]
        """
        self.shape = shape
        self.strength = strength

    def get_image(self):
        return np.random.randn(self.shape[0], self.shape[1], 3) * self.strength


class RandomImageSource(ImageSource):
    def __init__(self, shape, filelist, total_frames=None, grayscale=False):
        """
        Args:
            shape: [h, w]
            filelist: a list of image files
        """
        self.grayscale = grayscale
        self.total_frames = total_frames
        self.shape = shape
        self.filelist = filelist
        self.build_arr()
        self.current_idx = 0
        self.reset()

    def build_arr(self):
        self.total_frames = self.total_frames if self.total_frames else len(self.filelist)
        self.arr = np.zeros((self.total_frames, self.shape[0], self.shape[1]) + ((3,) if not self.grayscale else (1,)))
        for i in range(self.total_frames):
            # if i % len(self.filelist) == 0: random.shuffle(self.filelist)
            fname = self.filelist[i % len(self.filelist)]
            if self.grayscale: im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)[..., None]
            else:              im = cv2.imread(fname, cv2.IMREAD_COLOR)
            self.arr[i] = cv2.resize(im, (self.shape[1], self.shape[0])) ## THIS IS NOT A BUG! cv2 uses (width, height)

    def reset(self):
        self._loc = np.random.randint(0, self.total_frames)

    def get_image(self):
        return self.arr[self._loc]


class RandomVideoSource(ImageSource):
    def __init__(self, shape, filelist, random_bg=False, max_videos=20, grayscale=False):
        """
        Args:
            shape: [h, w]
            filelist: a list of video files
        """
        self.grayscale = grayscale
        self.shape = shape
        self.filelist = filelist
        random.shuffle(self.filelist)
        self.filelist = self.filelist[:max_videos]
        self.max_videos = max_videos
        self.random_bg = random_bg
        self.current_idx = 0
        self._current_vid = None
        self.reset()

    def load_video(self, vid_id):
        fname = self.filelist[vid_id]
        if self.grayscale:
            frames = skvideo.io.vread(fname, outputdict={"-pix_fmt": "gray"})
        else:
            frames = skvideo.io.vread(fname, num_frames=1000)
        img_arr = np.zeros((frames.shape[0], self.shape[0], self.shape[1]) + ((3,) if not self.grayscale else (1,)))
        for i in range(frames.shape[0]):
            img_arr[i] = cv2.resize(
                frames[i], (self.shape[1], self.shape[0])
            )  # THIS IS NOT A BUG! cv2 uses (width, height)
        return img_arr

    def reset(self):
        del self._current_vid
        while True:
            try:
                self._video_id = np.random.randint(0, len(self.filelist))
                self._current_vid = self.load_video(self._video_id)
                break
            except Exception:
                continue
        self._loc = np.random.randint(0, len(self._current_vid))

    def get_envindex(self):
        env_index = self._video_id
        return env_index


    def get_image(self):
        if self.random_bg:
            self._loc = np.random.randint(0, len(self._current_vid))
        else:
            self._loc += 1
        img = self._current_vid[self._loc % len(self._current_vid)]
        return img
