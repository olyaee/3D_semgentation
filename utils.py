import cv2
import imageio
from IPython.display import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import PIL
import SimpleITK as sitk
from augmentations import *

class VolumeDataGenerator(keras.utils.Sequence):
  def __init__(self,
              img_list,
              gt_list,
              batch_size=1,
              shuffle=True,
              in_ch=1,
              out_ch=9,
              crop = 'train_crop',
              dim_crop=(160, 160, 16), 
              bg_threshold=.9, 
              normalize=True,
              augment=True,
              to_categorical=True,
              channels_last=True,
              verbose=1):
    '''
    crop: one of the following optins. 'train_crop', 'test_crop', 'false'
    '''
    self.img_list = img_list
    self.gt_list = gt_list
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.in_ch = in_ch
    self.out_ch = out_ch
    self.id_list = np.arange(len(self.img_list))
    self.crop = crop
    self.dim_crop = dim_crop
    self.bg_threshold = bg_threshold
    self.verbose = verbose
    self.normalize = normalize
    self.augment = augment
    self.to_categorical = to_categorical
    self.channels_last = channels_last
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.img_list) / self.batch_size))

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    if self.shuffle == True:
      np.random.shuffle(self.id_list)

  def pad(self, vol):
    size = vol.GetSize()

    if self.crop is 'false':
      return vol
    
    elif self.crop is 'train_crop':
      max = [np.max((i, j)) for i, j in zip(size, self.dim_crop)]

    elif self.crop is 'test_crop':
      max = [((np.argwhere(np.arange(20)*16>dim)[0])*16).item() for dim in size]
      
    else:
      raise ValueError('crop option is no valid!')

    pad1 = [int(np.ceil((m-s)/2)) for m, s in zip(max, size)]
    pad2 = [int(np.floor((m-s)/2)) for m, s in zip(max, size)]

    return sitk.ConstantPad(vol, pad1 ,pad2, 0)


  def reset_sitk_params(self, img, gt):
    img = sitk.GetArrayFromImage(img)
    gt = sitk.GetArrayFromImage(gt)
    # in the dataset lablel of images are as follow
    # background:0, aorta:203, lv:204, pa:205, ra:206, svc:207, ivc:208, la:209, rv:210
    # and keras requires from 0 to 8
    gt[gt>0]-=202 
    return sitk.GetImageFromArray(img), sitk.GetImageFromArray(gt)


  def do_augment(self, img, gt):
    # add noise or smoothing
    SN = SmoothOrNoise(prob=0.5)
    img = SN.apply_transform(img)


    T = Translate(prob=0.5, t_min=-20, t_max=20).get_transform(img)
    S = Scale(prob=.5, min_sc=0.9, max_sc=1.1, same=True).get_transform(img)
    R = Rotate(prob_x=.5, prob_y=.5, prob_z=.5, alpha_min=-45, alpha_max=45).get_transform(img)
    F = Flip(prob_x=.5, prob_y=.5, prob_z=.5).get_transform(img)
    D = RadialDistortion(.3).get_transform(img)

    composite = sitk.Transform(3, sitk.sitkComposite)
    if T is not None: composite.AddTransform(T) 
    if S is not None: composite.AddTransform(S)
    if R is not None: composite.AddTransform(R)
    if F is not None: composite.AddTransform(F)
    if D is not None: composite.AddTransform(D)


    interp = sitk.sitkNearestNeighbor
    img_resampled = sitk.Resample(img, composite, interp, 0.0)
    gt_resampled = sitk.Resample(gt, composite, interp, 0.0)
    
    return img_resampled, gt_resampled

  def __data_generation(self, indexes):
    'Generates data containing batch_size samples'
    #set output channels
    if self.to_categorical:
      out_ch = self.out_ch 
    else: 
      out_ch = 1
    
    #initialize wrt channels position
    if self.crop is 'train_crop':
      if self.channels_last is True:
        input_shape = (self.batch_size, *self.dim_crop, self.in_ch)
        output_shape = (self.batch_size, *self.dim_crop, out_ch)
      else: 
        input_shape = (self.batch_size, self.in_ch, *self.dim_crop)
        output_shape = (self.batch_size, out_ch,  *self.dim_crop)
      X = np.zeros(input_shape, dtype=np.float64)
      y = np.zeros(output_shape, dtype=np.float64)

    for i, ID in enumerate(indexes):
      if self.verbose == 1:
        print("Training on: %s" % self.img_list[ID])
      
      #read img and ground truth
      img = sitk.ReadImage(self.img_list[ID])
      gt = sitk.ReadImage(self.gt_list[ID])

      img, gt = self.reset_sitk_params(img, gt)

      if self.normalize:
         img = self.z_normalize_img(img)

      # if self.crop is 'train_crop':
      img = self.pad(img)
      gt = self.pad(gt)
      img, gt = self.random_crop(img, gt, self.bg_threshold)

      if self.augment:
        img, gt = self.do_augment(img, gt)

      img = sitk.GetArrayFromImage(img)
      gt = sitk.GetArrayFromImage(gt)
      
      #sitk has an special dimension order and needs to be corrected
      img = np.moveaxis(img, [0, 1, 2], [2, 1, 0])
      gt = np.moveaxis(gt, [0, 1, 2], [2, 1, 0])

      if self.to_categorical:
        #to_categorical adds the channels to the last
        gt = keras.utils.to_categorical(gt, num_classes=9)
      else: 
        gt = np.expand_dims(gt, axis=-1)


      if self.channels_last is True:
        # print(img.shape)
        img = np.expand_dims(img, 3)
      else:
        img = np.expand_dims(img, 0)
        #move the channels of gt based on the CHANNLES_LAST
        gt = np.moveaxis(gt, -1, 0)

      # if the crop is False we can skip predefined output shape
      # note that if crop is false the batch size has be be 1
      if self.crop is 'false' or self.crop is 'test_crop':
        return np.expand_dims(img, 0), np.expand_dims(gt, 0)

      #insert the image and gt 
      X[i, ] = img
      y[i, ] = gt

    return X, y
                  
  def random_crop(self, img, gt, bg_threshold=.90, max_tries=15):
    if self.crop is 'false' or self.crop is 'test_crop':
      return img, gt

    img = sitk.GetArrayFromImage(img)
    gt = sitk.GetArrayFromImage(gt)

    orig_x, orig_y, orig_z = img.shape[:]
    output_x, output_y, output_z = self.dim_crop

    X = None
    y = None

    tries = 0

    while(tries<max_tries):
      start_x = np.random.randint(orig_x - output_x + 1) if orig_x>output_x else 0
      start_y = np.random.randint(orig_y - output_y + 1) if orig_y>output_y else 0
      start_z = np.random.randint(orig_z - output_z + 1) if orig_z>output_z else 0
      # sometimes image size is bigger than crop size and in this occation no crop is performed
      range_x = slice(start_x, start_x + output_x) if orig_x>output_x else slice(orig_x)
      range_y = slice(start_y, start_y + output_y) if orig_y>output_y else slice(orig_y)
      range_z = slice(start_z, start_z + output_z) if orig_z>output_z else slice(orig_z)

      # extract relevant area of label
      y = gt[range_x, range_y, range_z]
      temp = tf.keras.utils.to_categorical(y, num_classes=9)

      bgrd_ratio = np.sum(temp[:, :, :, 0])/(temp[:, :, :, 0]).size
      tries+=1
      print('try', tries, bgrd_ratio)
      if bgrd_ratio < bg_threshold:
        # make copy of the sub-volume
        X = np.copy(img[range_x, range_y, range_z])

        X = sitk.GetImageFromArray(X)
        y = sitk.GetImageFromArray(y)
        return X, y
        
    X = np.copy(img[range_x, range_y, range_z])
    X = sitk.GetImageFromArray(X)
    y = sitk.GetImageFromArray(y)
    return X, y
    
  def z_normalize_img(self, img):
    """
    Normalize the img so that the mean value for each img
    is 0 and the standard deviation is 1.
    """
    img = sitk.GetArrayFromImage(img)
    normalized_img = np.zeros(img.shape)

    for channel in range(img.shape[-1]):
      img_temp = img[:, :, channel]
      img_temp = (img_temp - np.mean(img_temp)) / np.std(img_temp)
      normalized_img[:, :, channel] = img_temp    
      
    normalized_img = sitk.GetImageFromArray(normalized_img)
    return normalized_img

  def __getitem__(self, index):
    'Generate one batch of data'
    indexes = self.id_list[
              index * self.batch_size: (index + 1) * self.batch_size]
    X, y = self.__data_generation(indexes)

    return X, y
    
    
def concat_h(imgs, mode='L'):
  'modes: L for gray images and RGB for colored images'
  shapes = [i.shape for i in imgs]
  w = ([i[1] for i in shapes])
  h = ([i[0] for i in shapes])
    
  if mode is 'L':
    imgs = [PIL.Image.fromarray((i*255).astype(np.uint8)) for i in imgs]
  elif mode is 'RGB':
    imgs = [PIL.Image.fromarray(cv2.normalize(i, None, alpha=0, beta=255,
                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(
                        np.uint8), mode = mode)
                        for i in imgs] 

  dst = PIL.Image.new(mode, (np.sum(w)+len(imgs)*5, np.max(h)), color=(128, 128, 128) if mode=='RGB' else 128)
  dst.paste(imgs[0], (0, 0))
  for i, img in enumerate(imgs[1:]):
    dst.paste(imgs[1:][i], (np.sum(w[:i+1])+5*(i+1), 0))
  return dst


def concat_v_pil(imgs, mode='L'):
  'modes: L for gray images and RGB for colored images'
  w = imgs[0].size[0]
  h = imgs[0].size[1]
  dst = PIL.Image.new(mode, (w, len(imgs)*(h+5)), color=(128, 128, 128) if mode=='RGB' else 128)
  dst.paste(imgs[0], (0, 0))
  for i, img in enumerate(imgs[1:]):
    dst.paste(imgs[1:][i], (0, h*(i+1)+5*(i+1)))
  return dst

def visualize_volume(img, gt, num_slices = 10):
  # images and gts
  img_rgb = np.stack((img, img, img), axis=3)
  coronal_img = np.flip(img_rgb, axis=2)
  axial_img = np.flip(img_rgb, axis=2)
  sagital_img = np.flip(img_rgb, axis=0)
  sagital_img = np.rot90(sagital_img, 3)

  coronal_gt = np.flip(gt, axis=2)
  axial_gt = np.flip(gt, axis=2)
  sagital_gt = np.flip(gt, axis=0)
  sagital_gt = np.rot90(sagital_gt, 3) 


  #colored gts
  colors = [(255.0/255, 51.0/255, 51.0/255), (255/255, 255/255, 51/255), (102/255, 204/255, 0/255), (51/255, 255/255, 255/255), (0, 128/255, 255/255), (127/255, 0, 255/255), (255/255, 0, 255/255), (204/255, 0, 102/255), (204/255, 0, 102/255)]
  coronal_gt_colored = np.zeros((*coronal_gt.shape, 3))
  axial_gt_colored = np.zeros((*axial_gt.shape, 3))
  sagital_gt_colored = np.zeros((*sagital_gt.shape, 3))
  for i in range (1, 9):
    coronal_gt_colored[coronal_gt==i] = colors[i-1]
    axial_gt_colored[axial_gt==i] = colors[i-1]
    sagital_gt_colored[sagital_gt==i] = colors[i-1]



  # colored gts added to imgs
  coronal_img_colored = coronal_img.copy()
  coronal_img_colored[coronal_gt>0] = 0
  coronal_img_colored = coronal_img_colored + coronal_gt_colored

  axial_img_colored = axial_img.copy()
  axial_img_colored[axial_gt>0] = 0
  axial_img_colored = axial_img_colored + axial_gt_colored
  # display(concat_h([coronal_img[i, :, :, :] for i in range(0, coronal_img.shape[0], coronal_img.shape[0]//num_slices)], mode='RGB'))

  sagital_img_colored = sagital_img.copy()
  sagital_img_colored[sagital_gt>0] = 0
  sagital_img_colored = sagital_img_colored + sagital_gt_colored

    
  out_coronal_llb = concat_h([coronal_gt_colored[i, :, :, :] for i in range(0, coronal_gt_colored.shape[0], coronal_gt_colored.shape[0]//num_slices)], mode='RGB') 
  out_coronal_img = concat_h([coronal_img[i, :, :, :] for i in range(0, coronal_img.shape[0], coronal_img.shape[0]//num_slices)], mode='RGB')
  out_coronal_comb = concat_h([coronal_img_colored[i, :, :, :] for i in range(0, coronal_img_colored.shape[0], coronal_img_colored.shape[0]//num_slices)], mode='RGB')


  out_axial_llb = concat_h([axial_gt_colored[:, i, :, :] for i in range(0, axial_gt_colored.shape[1], axial_gt_colored.shape[1]//num_slices)], mode='RGB')
  out_axial_img = concat_h([axial_img[:, i, :, :] for i in range(0, axial_img.shape[1], axial_img.shape[1]//num_slices)], mode='RGB')
  out_axial_comb = concat_h([axial_img_colored[:, i, :, :] for i in range(0, axial_img_colored.shape[1], axial_img_colored.shape[1]//num_slices)], mode='RGB')


  out_sagital_llb = concat_h([sagital_gt_colored[:, :, i, :] for i in range(0, sagital_gt_colored.shape[2], sagital_gt_colored.shape[2]//num_slices)], mode='RGB')
  out_sagital_img = concat_h([sagital_img[:, :, i, :] for i in range(0, sagital_img.shape[2], sagital_img.shape[2]//num_slices)], mode='RGB')
  out_sagital_comb = concat_h([sagital_img_colored[:, :, i, :] for i in range(0, sagital_img_colored.shape[2], sagital_img_colored.shape[2]//num_slices)], mode='RGB')
  
  #display(out_coronal_llb)
  #display(out_coronal_img)
  #display(out_coronal_comb)
  
  #display(out_axial_llb)
  #display(out_axial_img)
  #display(out_axial_comb)
  
  #display(out_sagital_llb)
  #display(out_sagital_img)
  #display(out_sagital_comb)
  
   
  out = [out_coronal_llb, out_coronal_img, out_axial_llb, out_axial_img, out_sagital_llb, out_sagital_img]

  out = concat_v_pil(out, mode='RGB')

  
  return out



def make_gif(img, gt):
  print('img.shape', img.shape)
  coronal_img = np.flip(img, axis=2)
  axial_img = np.flip(img, axis=2)
  sagital_img = np.flip(img, axis=0)
  sagital_img = np.rot90(sagital_img, 3)

  coronal_gt = np.flip(gt, axis=2)
  axial_gt = np.flip(gt, axis=2)
  sagital_gt = np.flip(gt, axis=0)
  sagital_gt = np.rot90(sagital_gt, 3) 

  output = []
  coronal_gif = []
  axial_gif = []
  sagital_gif = []

  coronal_gif = [np.array(concat_h((coronal_img[i, :, :], coronal_gt[i, :, :]))) for i in range(coronal_img.shape[0])]
  axial_gif = [np.array(concat_h((axial_img[:, i, :], axial_gt[:, i, :]))) for i in range(axial_img.shape[1])]
  sagital_gif = [np.array(concat_h((sagital_img[:, :, i], sagital_gt[:, :, i]))) for i in range(sagital_img.shape[2])]
  # coronal_gif = [cv2.normalize(coronal[i, :, :], None, alpha=0, beta=255,
  #                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(
  #                       np.uint8)
  #                       for i in range(coronal.shape[0])]
  # axial_gif = [cv2.normalize(axial[:, i, :], None, alpha=0, beta=255,
#                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(
  #                       np.uint8)
  #                       for i in range(axial.shape[1])]
  # sagital_gif = [cv2.normalize(sagital[:, :, i], None, alpha=0, beta=255,
#                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(
  #                       np.uint8)
  #                       for i in range(sagital.shape[2])]
  imageio.mimsave("/tmp/coronal.gif", coronal_gif, duration=0.01)
  output.append(Image(filename="/tmp/coronal.gif", format='png')) 
  imageio.mimsave("/tmp/axial.gif", axial_gif, duration=0.01)
  output.append(Image(filename="/tmp/axial.gif", format='png'))  
  imageio.mimsave("/tmp/sagital.gif", sagital_gif, duration=0.01)
  output.append(Image(filename="/tmp/sagital.gif", format='png'))
  return output

  # imgs = [coronal, axial, sagital]
  # dims =[coronal.shape[0], axial.shape[1], sagital.shape[2]]
  # print(dims)
  # idx = np.argsort(dims)
  # outputs =[]
  # for i in range(157):
  #   outputs.append(np.array(concat_h((imgs[0][i, :, :], imgs[1][:, i, :], imgs[2][:, :, i]))))

  # for j in range(157, 256):
  #   outputs.append(np.array(concat_h((coronal[j, :, :], axial[:, j, :], sagital[:, :, i]))))

  # for k in range(256, 273):
  #   outputs.append(np.array(concat_h((coronal[k, :, :], axial[:, j, :], sagital[:, :, i]))))

  # imageio.mimsave("/tmp/gif.gif", outputs, duration=0.01)
  # # del(outputs)
  # # return Image(filename="/tmp/gif.gif", format='png')



def undo_categorical(gt, channels_last=True):
  'img must be in the format of (b x y z c) or (b c x y z)'
  axis = -1 if channels_last is True else 0
  return np.argmax(gt[0, :, :, :, :], axis=axis)




def visualize_gt(gt, channels_last=True, num_slices = 10):
  # images and gts
  coronal_gt = np.flip(gt, axis=2)
  

  #for lbl in range(9):
  #  display(concat_h([coronal_gt[slc, :, :, lbl] for slc in range(0, coronal_gt.shape[0], coronal_gt.shape[0]//num_slices)], mode='L'))
  
  if channels_last is True:
    out = [concat_h([coronal_gt[slc, :, :, lbl] for slc in range(0, coronal_gt.shape[0], coronal_gt.shape[0]//num_slices)], mode='L') for lbl in range(9)]
  else:
    out = [concat_h([coronal_gt[lbl, slc, :, :] for slc in range(0, coronal_gt.shape[1], coronal_gt.shape[1]//num_slices)], mode='L') for lbl in range(9)]
  
  out = concat_v_pil(out, mode='L')
  return out
  