import numpy as np
import SimpleITK as sitk


class Translate:
    #coordinate space
    def __init__(self,prob=0.5, t_min=-10, t_max=10):
        self.t_min= t_min
        self.t_max= t_max
        self.prob= prob
        self.name= 'Translate'

    def get_transform(self, volume):
        translation=None
        probability = np.random.uniform(0, 1)
        tx = np.random.uniform(self.t_min, self.t_max)
        ty = np.random.uniform(self.t_min, self.t_max)
        tz = np.random.uniform(self.t_min, self.t_max)
        if (probability < self.prob):
            translation = sitk.TranslationTransform(3, np.array([tx,ty,tz]))
        return translation

class Scale:

    def __init__(self, prob=.5, min_sc=0.9, max_sc=1.1, same=True):

        self.min_sc=min_sc
        self.max_sc=max_sc
        self.prob=prob
        self.same= same
        self.name='Scale'
    def get_transform(self, volume):
        probability= np.random.uniform(0,1)
        scale= None
        if(probability< self.prob):
            if self.same :
                scx=scy=scz= np.random.uniform(self.min_sc, self.max_sc)
            else:
                scx, scy, scz=np.random.uniform(self.min_sc, self.max_sc, size=(1,3))
            sc= (scx, scy, scz)
            scale = sitk.ScaleTransform(3, sc)
            center = np.array(volume.TransformContinuousIndexToPhysicalPoint(np.array(volume.GetSize())/2.0))
            scale.SetCenter(center)
        return scale

class Rotate:
    def __init__(self,  prob_x=.5, prob_y=.5, prob_z=.5, alpha_min=-45, alpha_max=45 ):
        self.alpha_min= alpha_min
        self.alpha_max= alpha_max
        self.prob_x= prob_x
        self.prob_y= prob_y
        self.prob_z= prob_z

        self.name= 'Rotate'

    def get_transform(self, volume):
        volume_shape= np.array(volume.GetSize())

        rotation= None


        rotate_x = 1 if np.random.uniform(0, 1)< self.prob_x else 0
        rotate_y = 1 if np.random.uniform(0, 1)< self.prob_y else 0
        rotate_z = 1 if np.random.uniform(0, 1)< self.prob_z else 0



        theta_x= np.random.uniform(self.alpha_min, self.alpha_max)  if rotate_x else 0
        theta_y= np.random.uniform(self.alpha_min, self.alpha_max) if rotate_y else 0
        theta_z= np.random.uniform(self.alpha_min, self.alpha_max) if rotate_z else 0

            
        # theta_x=0*(np.pi/180)
        # theta_y=0*(np.pi/180)
        # theta_z=20*(np.pi/180)

        translation = (0, 0, 0)
        # rotation_center = (int(volume_shape[0] / 2), int(volume_shape[1] / 2), int(volume_shape[2] / 2))
        rotation_center = np.array(volume.TransformContinuousIndexToPhysicalPoint(np.array(volume.GetSize())/2.0))
        rotation = sitk.Euler3DTransform(rotation_center, theta_x, theta_y, theta_z, translation)

        return rotation


class Flip:
    def __init__(self, prob_x=.5, prob_y=.5, prob_z=.5):

        self.prob_x= prob_x
        self.prob_y= prob_y
        self.prob_z= prob_z
        self.name= 'Flip'

    def get_transform(self, volume):
        flip = sitk.AffineTransform(3)

        flip_x = -1 if np.random.uniform(0, 1)< self.prob_x else 1
        flip_y = -1 if np.random.uniform(0, 1)< self.prob_x else 1
        flip_z = -1 if np.random.uniform(0, 1)< self.prob_x else 1
        flip.SetMatrix([flip_x,0,0,0,flip_y,0,0,0,flip_z])
    
        volume_shape= np.array(volume.GetSize())
        # center = (int(volume_shape[0] / 2), int(volume_shape[1] / 2), int(volume_shape[2] / 2))
        center = np.array(volume.TransformContinuousIndexToPhysicalPoint(np.array(volume.GetSize())/2.0))
        flip.SetCenter(center)
        return flip

    def get_axis(self):
        return self.axis




class RadialDistortion:
  def __init__(self, prob=.3, k1_range=(1e-9, 1e-6), k2_range=(1e-14, 1e-12), k3_range=(1e-14, 1e-12), center_range=(-15, 15)):
    '''
    center_range is the distance from the center of image that could be transformation center
    '''
    self.prob = prob
    self.center_range=center_range

    self.k1_min = k1_range[0]
    self.k1_max = k1_range[1]

    self.k2_min = k2_range[0]
    self.k2_max = k2_range[1]

    self.k3_min = k3_range[0]
    self.k3_max = k3_range[1]

  def get_transform(self, volume):

    displacement_field_transform = None

    if np.random.uniform(0, 1)< self.prob:

      self.k1 = np.random.uniform(self.k1_min, self.k1_max)
      self.k2 = np.random.uniform(self.k2_min, self.k2_max)
      self.k3 = np.random.uniform(self.k3_min, self.k3_max)

      # The default distortion center coincides with the volume center
      c = np.array(volume.TransformContinuousIndexToPhysicalPoint(np.array(volume.GetSize())/2.0))
      c = [ i+np.random.uniform(self.center_range[0], self.center_range[1]) for i in c]
      
      # Compute the vector volume (p_d - p_c) 
      delta_volume = sitk.PhysicalPointSource( sitk.sitkVectorFloat64, volume.GetSize(), volume.GetOrigin(), volume.GetSpacing(), volume.GetDirection())
      delta_volume_list = [sitk.VectorIndexSelectionCast(delta_volume,i) - c[i] for i in range(len(c))]
      
      # Compute the radial distortion expression
      r2_volume = sitk.NaryAdd([vol**2 for vol in delta_volume_list])
      r4_volume = r2_volume**2
      r6_volume = r2_volume*r4_volume
      disp_volume = self.k1*r2_volume + self.k2*r4_volume + self.k3*r6_volume
      displacement_volume = sitk.Compose([disp_volume*vol for vol in delta_volume_list])
      
      displacement_field_transform = sitk.DisplacementFieldTransform(displacement_volume)
      
    return displacement_field_transform

class SmoothOrNoise():
  def __init__(self, prob=0.6):
    self.prob=prob

  def apply_transform(self, volume):

    if np.random.uniform(0, 1)< self.prob:
      smoothornoise = np.random.randint(2)

      if smoothornoise==0:

        # Smoothing filters
        rand = np.random.randint(3)

        if rand==0:
          f = sitk.SmoothingRecursiveGaussianImageFilter()
          s = np.random.uniform(2.0)
          f.SetSigma(s)

        elif rand==1:  
          f = sitk.DiscreteGaussianImageFilter()
          v = np.random.uniform(4.0)
          f.SetVariance(v)

        elif rand==2:
          f = sitk.MedianImageFilter()
          r = np.random.randint(8)
          f.SetRadius(r)

        # #its not working
        # elif rand==3:
        #   f = sitk.BilateralImageFilter()
        #   ds = np.random.uniform(4.0)
        #   rs = np.random.uniform(8.0)
        #   f.SetDomainSigma(ds)
        #   f.SetRangeSigma(rs)

      elif smoothornoise==1:
        rand = np.random.randint(3)

        if rand==0:
          f = sitk.AdditiveGaussianNoiseImageFilter()
          f.SetMean(0)
          f.SetStandardDeviation(np.random.uniform(0, .3))
        
        elif rand==1:
          f = sitk.ShotNoiseImageFilter()
          f.SetScale(np.random.uniform(10, 30))

        elif rand==2:
          f = sitk.SpeckleNoiseImageFilter()
          f.SetStandardDeviation(np.random.uniform(.1, .2))

        # #its not working
        # elif rand==3:
        #   f = sitk.SaltAndPepperNoiseImageFilter()
        #   f.SetProbability(0.001)

        # AdaptiveHistogramEqualizationImageFilter is very time consuming
        # elif rand==4:
        #   f = sitk.AdaptiveHistogramEqualizationImageFilter()
        #   f.SetAlpha(1.0)
        #   f.SetBeta(0.0)

        # elif rand==5:
        #   f = sitk.AdaptiveHistogramEqualizationImageFilter()
        #   f.SetAlpha(1.0)
        #   f.SetBeta(0.0)

      return f.Execute(volume)
    else:
      return volume


              


