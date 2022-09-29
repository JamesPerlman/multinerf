# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Different datasets implementation plus a general port for all the datasets."""

import abc
import copy
import json
import os
from os import path
from pathlib import Path
from plistlib import FMT_XML
import queue
import threading
from typing import Mapping, Optional, Sequence, Text, Tuple, Union

import cv2
from internal import camera_utils
from internal import configs
from internal import image as lib_image
from internal import raw_utils
from internal import utils
import jax
import numpy as np
from PIL import Image

# This is ugly, but it works.
import sys
sys.path.insert(0,'internal/pycolmap')
sys.path.insert(0,'internal/pycolmap/pycolmap')
import pycolmap


def load_dataset(split, train_dir, config):
  """Loads a split of a dataset using the data_loader specified by `config`."""
  return JPDataset(split, train_dir, config)

class Dataset(threading.Thread, metaclass=abc.ABCMeta):
  """Dataset Base Class.

  Base class for a NeRF dataset. Creates batches of ray and color data used for
  training or rendering a NeRF model.

  Each subclass is responsible for loading images and camera poses from disk by
  implementing the _load_renderings() method. This data is used to generate
  train and test batches of ray + color data for feeding through the NeRF model.
  The ray parameters are calculated in _generate_rays().

  The public interface mimics the behavior of a standard machine learning
  pipeline dataset provider that can provide infinite batches of data to the
  training/testing pipelines without exposing any details of how the batches are
  loaded/created or how this is parallelized. Therefore, the initializer runs
  all setup, including data loading from disk using _load_renderings(), and
  begins the thread using its parent start() method. After the initializer
  returns, the caller can request batches of data straight away.

  The internal self._queue is initialized as queue.Queue(3), so the infinite
  loop in run() will block on the call self._queue.put(self._next_fn()) once
  there are 3 elements. The main thread training job runs in a loop that pops 1
  element at a time off the front of the queue. The Dataset thread's run() loop
  will populate the queue with 3 elements, then wait until a batch has been
  removed and push one more onto the end.

  This repeats indefinitely until the main thread's training loop completes
  (typically hundreds of thousands of iterations), then the main thread will
  exit and the Dataset thread will automatically be killed since it is a daemon.

  Attributes:
    alphas: np.ndarray, optional array of alpha channel data.
    cameras: tuple summarizing all camera extrinsic/intrinsic/distortion params.
    camtoworlds: np.ndarray, a list of extrinsic camera pose matrices.
    camtype: camera_utils.ProjectionType, fisheye or perspective camera.
    data_dir: str, location of the dataset on disk.
    disp_images: np.ndarray, optional array of disparity (inverse depth) data.
    distortion_params: dict, the camera distortion model parameters.
    exposures: optional per-image exposure value (shutter * ISO / 1000).
    far: float, far plane value for rays.
    focal: float, focal length from camera intrinsics.
    height: int, height of images.
    images: np.ndarray, array of RGB image data.
    metadata: dict, optional metadata for raw datasets.
    near: float, near plane value for rays.
    normal_images: np.ndarray, optional array of surface normal vector data.
    pixtocams: np.ndarray, one or a list of inverse intrinsic camera matrices.
    pixtocam_ndc: np.ndarray, the inverse intrinsic matrix used for NDC space.
    poses: np.ndarray, optional array of auxiliary camera pose data.
    rays: utils.Rays, ray data for every pixel in the dataset.
    render_exposures: optional list of exposure values for the render path.
    render_path: bool, indicates if a smooth camera path should be generated.
    size: int, number of images in the dataset.
    split: str, indicates if this is a "train" or "test" dataset.
    width: int, width of images.
  """

  def __init__(self,
               split: str,
               data_dir: str,
               config: configs.Config):
    super().__init__()

    # Initialize attributes
    self._queue = queue.Queue(3)  # Set prefetch buffer to 3 batches.
    self.daemon = True  # Sets parent Thread to be a daemon.
    self._patch_size = np.maximum(config.patch_size, 1)
    self._batch_size = config.batch_size // jax.process_count()
    if self._patch_size**2 > self._batch_size:
      raise ValueError(f'Patch size {self._patch_size}^2 too large for ' +
                       f'per-process batch size {self._batch_size}')
    self._batching = utils.BatchingMethod(config.batching)
    self._use_tiffs = config.use_tiffs
    self._load_disps = config.compute_disp_metrics
    self._load_normals = config.compute_normal_metrics
    self._test_camera_idx = 0
    self._num_border_pixels_to_mask = config.num_border_pixels_to_mask
    self._apply_bayer_mask = config.apply_bayer_mask
    self._cast_rays_in_train_step = config.cast_rays_in_train_step
    self._render_spherical = False

    self.split = utils.DataSplit(split)
    self.data_dir = data_dir
    self.near = config.near
    self.far = config.far
    self.render_path = config.render_path
    self.distortion_params = None
    self.disp_images = None
    self.normal_images = None
    self.alphas = None
    self.poses = None
    self.pixtocam_ndc = None
    self.metadata = None
    self.camtype = camera_utils.ProjectionType.PERSPECTIVE
    self.exposures = None
    self.render_exposures = None

    # Providing type comments for these attributes, they must be correctly
    # initialized by _load_renderings() (see docstring) in any subclass.
    self.images: np.ndarray = None
    self.camtoworlds: np.ndarray = None
    self.pixtocams: np.ndarray = None
    self.height: int = None
    self.width: int = None

    # Load data from disk using provided config parameters.
    self._load_renderings(config)

    if self.render_path:
      if config.render_path_file is not None:
        with utils.open_file(config.render_path_file, 'rb') as fp:
          render_poses = np.load(fp)
        self.camtoworlds = render_poses
      if config.render_resolution is not None:
        self.width, self.height = config.render_resolution
      if config.render_focal is not None:
        self.focal = config.render_focal
      if config.render_camtype is not None:
        if config.render_camtype == 'pano':
          self._render_spherical = True
        else:
          self.camtype = camera_utils.ProjectionType(config.render_camtype)

      self.distortion_params = None
      self.pixtocams = camera_utils.get_pixtocam(self.focal, self.width,
                                                 self.height)

    self._n_examples = self.camtoworlds.shape[0]

    self.cameras = (self.pixtocams,
                    self.camtoworlds,
                    self.distortion_params,
                    self.pixtocam_ndc)

    # Seed the queue with one batch to avoid race condition.
    if self.split == utils.DataSplit.TRAIN:
      self._next_fn = self._next_train
    else:
      self._next_fn = self._next_test
    self._queue.put(self._next_fn())
    self.start()

  def __iter__(self):
    return self

  def __next__(self):
    """Get the next training batch or test example.

    Returns:
      batch: dict, has 'rgb' and 'rays'.
    """
    x = self._queue.get()
    if self.split == utils.DataSplit.TRAIN:
      return utils.shard(x)
    else:
      # Do NOT move test `rays` to device, since it may be very large.
      return x

  def peek(self):
    """Peek at the next training batch or test example without dequeuing it.

    Returns:
      batch: dict, has 'rgb' and 'rays'.
    """
    x = copy.copy(self._queue.queue[0])  # Make a copy of front of queue.
    if self.split == utils.DataSplit.TRAIN:
      return utils.shard(x)
    else:
      return jax.device_put(x)

  def run(self):
    while True:
      self._queue.put(self._next_fn())

  @property
  def size(self):
    return self._n_examples

  @abc.abstractmethod
  def _load_renderings(self, config):
    """Load images and poses from disk.

    Args:
      config: utils.Config, user-specified config parameters.
    In inherited classes, this method must set the following public attributes:
      images: [N, height, width, 3] array for RGB images.
      disp_images: [N, height, width] array for depth data (optional).
      normal_images: [N, height, width, 3] array for normals (optional).
      camtoworlds: [N, 3, 4] array of extrinsic pose matrices.
      poses: [..., 3, 4] array of auxiliary pose data (optional).
      pixtocams: [N, 3, 4] array of inverse intrinsic matrices.
      distortion_params: dict, camera lens distortion model parameters.
      height: int, height of images.
      width: int, width of images.
      focal: float, focal length to use for ideal pinhole rendering.
    """

  def _make_ray_batch(self,
                      pix_x_int: np.ndarray,
                      pix_y_int: np.ndarray,
                      cam_idx: Union[np.ndarray, np.int32],
                      lossmult: Optional[np.ndarray] = None
                      ) -> utils.Batch:
    """Creates ray data batch from pixel coordinates and camera indices.

    All arguments must have broadcastable shapes. If the arguments together
    broadcast to a shape [a, b, c, ..., z] then the returned utils.Rays object
    will have array attributes with shape [a, b, c, ..., z, N], where N=3 for
    3D vectors and N=1 for per-ray scalar attributes.

    Args:
      pix_x_int: int array, x coordinates of image pixels.
      pix_y_int: int array, y coordinates of image pixels.
      cam_idx: int or int array, camera indices.
      lossmult: float array, weight to apply to each ray when computing loss fn.

    Returns:
      A dict mapping from strings utils.Rays or arrays of image data.
      This is the batch provided for one NeRF train or test iteration.
    """
    broadcast_scalar = lambda x: np.broadcast_to(x, pix_x_int.shape)[..., None]
    ray_kwargs = {
        'lossmult': broadcast_scalar(1.) if lossmult is None else lossmult,
        'near': broadcast_scalar(self.nears[cam_idx]),
        'far': broadcast_scalar(self.fars[cam_idx]),
        'cam_idx': broadcast_scalar(cam_idx),
    }
    # Collect per-camera information needed for each ray.
    if self.metadata is not None:
      # Exposure index and relative shutter speed, needed for RawNeRF.
      for key in ['exposure_idx', 'exposure_values']:
        idx = 0 if self.render_path else cam_idx
        ray_kwargs[key] = broadcast_scalar(self.metadata[key][idx])
    if self.exposures is not None:
      idx = 0 if self.render_path else cam_idx
      ray_kwargs['exposure_values'] = broadcast_scalar(self.exposures[idx])
    if self.render_path and self.render_exposures is not None:
      ray_kwargs['exposure_values'] = broadcast_scalar(
          self.render_exposures[cam_idx])

    pixels = utils.Pixels(pix_x_int, pix_y_int, **ray_kwargs)
    if self._cast_rays_in_train_step and self.split == utils.DataSplit.TRAIN:
      # Fast path, defer ray computation to the training loop (on device).
      rays = pixels
    else:
      # Slow path, do ray computation using numpy (on CPU).
      rays = camera_utils.cast_ray_batch(
          self.cameras, pixels, self.camtype, xnp=np)

    # Create data batch.
    batch = {}
    batch['rays'] = rays
    if not self.render_path:
      batch['rgb'] = self.images[cam_idx, pix_y_int, pix_x_int]
    if self._load_disps:
      batch['disps'] = self.disp_images[cam_idx, pix_y_int, pix_x_int]
    if self._load_normals:
      batch['normals'] = self.normal_images[cam_idx, pix_y_int, pix_x_int]
      batch['alphas'] = self.alphas[cam_idx, pix_y_int, pix_x_int]
    return utils.Batch(**batch)

  def _next_train(self) -> utils.Batch:
    """Sample next training batch (random rays)."""
    # We assume all images in the dataset are the same resolution, so we can use
    # the same width/height for sampling all pixels coordinates in the batch.
    # Batch/patch sampling parameters.
    num_patches = self._batch_size // self._patch_size ** 2
    lower_border = self._num_border_pixels_to_mask
    upper_border = self._num_border_pixels_to_mask + self._patch_size - 1
    # Random pixel patch x-coordinates.
    pix_x_int = np.random.randint(lower_border, self.width - upper_border,
                                  (num_patches, 1, 1))
    # Random pixel patch y-coordinates.
    pix_y_int = np.random.randint(lower_border, self.height - upper_border,
                                  (num_patches, 1, 1))
    # Add patch coordinate offsets.
    # Shape will broadcast to (num_patches, _patch_size, _patch_size).
    patch_dx_int, patch_dy_int = camera_utils.pixel_coordinates(
        self._patch_size, self._patch_size)
    pix_x_int = pix_x_int + patch_dx_int
    pix_y_int = pix_y_int + patch_dy_int
    # Random camera indices.
    if self._batching == utils.BatchingMethod.ALL_IMAGES:
      cam_idx = np.random.randint(0, self._n_examples, (num_patches, 1, 1))
    else:
      cam_idx = np.random.randint(0, self._n_examples, (1,))

    if self._apply_bayer_mask:
      # Compute the Bayer mosaic mask for each pixel in the batch.
      lossmult = raw_utils.pixels_to_bayer_mask(pix_x_int, pix_y_int)
    else:
      lossmult = None

    return self._make_ray_batch(pix_x_int, pix_y_int, cam_idx,
                                lossmult=lossmult)

  def generate_ray_batch(self, cam_idx: int) -> utils.Batch:
    """Generate ray batch for a specified camera in the dataset."""
    if self._render_spherical:
      camtoworld = self.camtoworlds[cam_idx]
      rays = camera_utils.cast_spherical_rays(
          camtoworld, self.height, self.width, self.near, self.far, xnp=np)
      return utils.Batch(rays=rays)
    else:
      # Generate rays for all pixels in the image.
      pix_x_int, pix_y_int = camera_utils.pixel_coordinates(
          self.width, self.height)
      return self._make_ray_batch(pix_x_int, pix_y_int, cam_idx)

  def _next_test(self) -> utils.Batch:
    """Sample next test batch (one full image)."""
    # Use the next camera index.
    cam_idx = self._test_camera_idx
    self._test_camera_idx = (self._test_camera_idx + 1) % self._n_examples
    return self.generate_ray_batch(cam_idx)


class JPDataset(Dataset):
  """jperl's Blender NeRF Tools Dataset."""

  def _parse_transforms(self, transforms_path: str, scale_factor: int = 1):

    data: dict
    with open(transforms_path, 'r') as f:
        data = json.load(f)

    n_images = len(data['frames'])

    w = int(data['w'])
    h = int(data['h'])
    fl = data['fl_x']

    if scale_factor > 1:
      w = int(w / scale_factor)
      h = int(h / scale_factor)
      fl = fl / scale_factor

    camtoworlds = []
    pixtocams = []
    image_paths = []

    # Loop over all images.
    for i in range(n_images):
      frame = data['frames'][i]

      image_paths.append(frame["file_path"])
      
      pose = np.array(frame['transform_matrix'])

      if scale_factor > 1:
        pose = np.diag([1. / scale_factor, 1. / scale_factor, 1. / scale_factor, 1.]).astype(np.float32) @ pose

      camtoworlds.append(pose)
      
      cx: float
      if 'camera_angle_x' in frame:
        cx = frame['camera_angle_x']
      else:
        cx = data['camera_angle_x']
      
      cam_fl = 0.5 * w / np.tan(0.5 * cx)
      p2c = camera_utils.get_pixtocam(cam_fl, w, h)
      pixtocams.append(p2c)

    
    nears = np.array([f['near'] for f in data['frames']])
    fars = np.array([f['far'] for f in data['frames']])

    coeffs = ['k1', 'k2', 'p1', 'p2']
    params = {c: (data[c] if c in data else 0.) for c in coeffs}

    return (image_paths, camtoworlds, pixtocams, fl, w, h, params, nears, fars)

  def _load_renderings(self, config):
    """Load images from disk."""

    fname: str
    
    if config.render_path:
      self.render_path = True
      fname = 'render.json'
    else:
      fname = 'transforms.json'
    
    transforms_path = os.path.join(self.data_dir, fname)
    
    (image_paths, camtoworlds, pixtocams, focal, width, height, distortion_params, nears, fars) = self._parse_transforms(
      transforms_path,
      config.factor
    )

    if config.rawnerf_mode:
      # Load raw images and metadata.
      images, metadata, raw_testscene = raw_utils.load_raw_dataset(
          self.split,
          self.data_dir,
          [Path(image_path).name for image_path in image_paths],
          config.exposure_percentile,
          config.factor)
      self.metadata = metadata
    elif not self.render_path:
      images = []

      # Load image.
      for image_path in image_paths:
        image = utils.load_img(os.path.join(self.data_dir, image_path)) / 255.
    
        if config.factor > 1:
          image = lib_image.downsample(image, config.factor)
        
        images.append(image)
        self.images = np.stack(images)
    
    camtoworlds = np.stack(camtoworlds, axis=0)

    self._n_examples = len(camtoworlds)
    self.focal = focal
    self.height = height
    self.width = width
    self.camtoworlds = camtoworlds
    self.pixtocams = np.stack(pixtocams)
    self.distortion_params = distortion_params
    self.nears = nears
    self.fars = fars

    if self.split == 'test':
      self.images = []
      self.camtoworlds = []
      self.pixtocams = []
      self._n_examples = 0
