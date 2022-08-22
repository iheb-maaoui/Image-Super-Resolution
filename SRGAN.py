import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg19 import VGG19,preprocess_input
from keras.layers import Dense,Conv2D,BatchNormalization,add,Input,UpSampling2D,LeakyReLU,Flatten
from keras.layers import PReLU
from keras.models import Model
from keras.losses import BinaryCrossentropy,MeanSquaredError,MeanAbsoluteError
from keras.optimizers import Adam

class Generator(tf.keras.models.Model):
  def __init__(self,lr_input,*args,**kwargs):
    super().__init__(self,*args,**kwargs)
    self.gen_input = lr_input
  
  def create_generator(self,res_blocks_nb):
    x = Conv2D(64,(9,9),1,padding='same')(self.gen_input)
    x = PReLU(shared_axes=[1,2])(x)

    aux = x
    for i in range(res_blocks_nb):
      x = self.residual_block(x)
        
    x = Conv2D(64,(3,3),1,padding='same')(x)
    x = BatchNormalization()(x)
    x = add([aux,x])

    x = self.upscaling_block(x)
    x = self.upscaling_block(x)

    x = Conv2D(3,(9,9),1,padding='same',activation='sigmoid')(x)

    return Model(self.gen_input,x,name='Generator')

  def residual_block(self,layer):
    x = Conv2D(64,(3,3),1,padding='same')(layer)
    x = BatchNormalization()(x)
    x = PReLU(shared_axes=[1,2])(x)
    x = Conv2D(64,(3,3),1,padding='same')(x)
    x = BatchNormalization()(x)
    return add([layer,x])
  
  def upscaling_block(self,layer):
    x = Conv2D(256,(3,3),1, padding='same')(layer)
    x = UpSampling2D(size = 2)(x)
    x = PReLU(shared_axes=[1,2])(x)

    return x


class Discriminator(tf.keras.models.Model):
  def __init__(self,gen_img_shape,*args,**kwargs):
    super().__init__(self,*args,**kwargs)
    self.shape = gen_img_shape

  def create_discriminator(self):
    input = Input(shape = self.shape)   
    x = Conv2D(64,(3,3),1,padding='same')(input)
    x = LeakyReLU(alpha=0.2)(x)

    for n in [1,2,4]:
      for s in [1,2]:
        x = self.disc_bloc(x,n,s)
    
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha = 0.2)(x)
    x = Dense(1,'sigmoid')(x)

    return Model(input,x,name='discriminator')

  def disc_bloc(self,layer,n,s):
    x = Conv2D(128*n,(3,3),s,padding='same')(layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    return x


from keras.engine import training
class SuperResolution(tf.keras.models.Model):
  def __init__(self,low_res_image_shape,high_res_image_shape,gen_res_blocks,*args,**kwargs):
    super().__init__(self,*args,**kwargs)
    self.lr_input = Input(shape=low_res_image_shape)
    self.hr_input = Input(shape=high_res_image_shape)
    
    self.generator = Generator(self.lr_input).create_generator(gen_res_blocks)
    self.discriminator = Discriminator(high_res_image_shape).create_discriminator()

    vgg = VGG19(include_top = False)
    self.vgg = Model(vgg.input,vgg.layers[-10].output,name='feature_extractor')

    self.model = self.merge_gen_disc()

  def merge_gen_disc(self):

    gen_img = self.generator.output

    gen_features = self.vgg(gen_img)

    disc_guess = self.discriminator(gen_img)

    return Model(inputs=[self.lr_input,self.hr_input],outputs=[disc_guess,gen_features]) 

  def compile(self,*args,**kwargs):
    super().compile(*args,**kwargs)
    self.g_opt = Adam(learning_rate=0.0001)
    self.d_opt = Adam(learning_rate=0.00001)
    self.adversarial_loss = BinaryCrossentropy()
    self.vgg_loss = MeanSquaredError()
    self.d_loss = BinaryCrossentropy()
    self.image_loss = MeanAbsoluteError()
  
  def train_step(self,batch):

    (img_lrs,img_hrs),real_label = batch
    real_label = tf.convert_to_tensor(real_label) 
    img_hrs = img_hrs/255.0
    img_lrs = img_lrs/255.0
    feature_hrs = self.vgg(img_hrs)
    fake_label = tf.zeros_like(real_label) + 0.15 * tf.random.uniform(shape=(1,),dtype=tf.float64)
    real_label = real_label #- 0.15 * tf.random.normal(shape=(1,),dtype=tf.float64)
    img_fakes = self.generator(img_lrs)

    self.discriminator.trainable = True

    with tf.GradientTape() as d_tape:
      ypred_real = self.discriminator(img_hrs,training=True)
      ypred_fake = self.discriminator(img_fakes,training=True)
      ypred = tf.concat([ypred_real,ypred_fake],axis=0)
      y_gt = tf.concat([real_label,fake_label],axis=0)
      total_d_loss = self.d_loss(y_gt,ypred)
    d_grad = d_tape.gradient(total_d_loss,self.discriminator.trainable_variables)
    self.d_opt.apply_gradients(zip(d_grad,self.discriminator.trainable_variables))

    self.discriminator.trainable = False
    with tf.GradientTape() as g_tape:
      [disc_guess,gen_features] = self.model([img_lrs,img_hrs],training=True)
      img_fakes = self.generator(img_lrs,training=True)
      y_gt = tf.ones_like(disc_guess)
      total_g_loss = 0.06 * self.image_loss(img_hrs,img_fakes) + 0.001 * self.adversarial_loss(y_gt,disc_guess) + self.vgg_loss(gen_features,feature_hrs) 
    g_grad = g_tape.gradient(total_g_loss,self.generator.trainable_variables)
    self.g_opt.apply_gradients(zip(g_grad,self.generator.trainable_variables))

    return {'g_loss':total_g_loss,'d_loss':total_d_loss}