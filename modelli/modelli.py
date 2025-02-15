import tensorflow as tf


class Autoencoder(tf.keras.models.Model):
  
  def __init__(self,mode=2):
    """
    mode: 2 for halving images, 4 for quartering
    """
    if mode==2 :
      self.sizeRes=50
    elif mode==4:
      self.sizeRes=25
    else:
      raise ValueError("mode must be 2 or 4")
    
    self.mode=mode
    super(Autoencoder, self).__init__()
    encoderModel=self.encoderCreate()
    decoderModel=self.decoderCreate()
    self.encoder = tf.keras.Model(encoderModel[0],encoderModel[1])
    self.decoder = tf.keras.Model(decoderModel[0],decoderModel[1])
  
  def encoderCreate(self):
    inputLayer=tf.keras.layers.Input(shape=(100, 100, 3),name="input")
    #resizingLayer=tf.keras.layers.Resizing(height=50,width=50)(inputLayer)
    #layerGreyScale = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x * tf.constant([0.21, 0.72, 0.07], dtype=tf.float32), axis=-1, keepdims=True))(resizingLayer)
    layerGreyScale=tf.keras.layers.Conv2D(3,(3,3),activation='relu', padding='same', strides=2)(inputLayer)
    ourput2=tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid', padding='same', strides=1)(layerGreyScale)
    
    resizingLayerStandard=tf.keras.layers.Resizing(height=100,width=100)(inputLayer)
    final=self.encoderRGBlsyeer(resizingLayerStandard)
    return inputLayer,[ourput2,final]
  

  def encoderRGBlsyeer(self,input):
    convLayer=tf.keras.layers.Conv2D(3, (3,3), activation='sigmoid', padding='same', strides=2)(input)
    if self.mode==2:
      return convLayer
    elif self.mode==4:
        return tf.keras.layers.Conv2D(3, (3,3), activation='sigmoid', padding='same', strides=2)(convLayer)
    

    
  def decoderRGBlayer(self,input):
    def invBlock(inputF):
      invConv=tf.keras.layers.Conv2DTranspose(5, (3, 3), strides=(2, 2), padding="same")(inputF)
      conv1=tf.keras.layers.Conv2D(5, (5, 5),strides=1, padding="same")(invConv)
      conv2=tf.keras.layers.Conv2D(3, (3, 3), strides=1, padding="same")(conv1)
      return conv2
    if self.mode==2 :
        return invBlock(inputF=input) 
    if self.mode==4:
      
      return invBlock(invBlock(inputF=input))
  
 
  def decoderCreate(self):
    input1=tf.keras.layers.Input(shape=(50, 50, 1))
    scalinImage=tf.keras.layers.UpSampling2D(2)(input1)
    conv=tf.keras.layers.Conv2D(1, (3, 3),activation='relu', strides=1, padding="same")(scalinImage)
    input2=tf.keras.layers.Input(shape=(self.sizeRes, self.sizeRes, 3))
    rgbLayers=self.decoderRGBlayer(input2)
    merged=tf.keras.layers.Add()([rgbLayers,conv])
    fianl=tf.keras.layers.Conv2D(3, (3, 3), strides=1, padding="same")(merged)

    return  [input1,input2],fianl
  

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)

    return decoded


def ssim_loss(y_true,y_pred ):
  result=0
  for i in range(0,3):
      result+=tf.reduce_mean(tf.image.ssim(y_true[:][:][i], y_pred[:][:][i], 1.0))
  return 1.0-result/3.0

def costum_loss(y_true,y_pred ):
  loss=tf.losses.mse(y_true,y_pred)
  ssim_cal=ssim_loss(y_true,y_pred)
  return loss+ssim_cal*loss


class AutoencoderFinal(tf.keras.Model):

  def __init__(self, oldAutoencoder,mode=2):
    super(AutoencoderFinal, self).__init__()
    encoderOld=oldAutoencoder.encoder
    decoderOld=oldAutoencoder.decoder
    for layer in encoderOld.layers:
        layer.trainable=False
    self.mode=mode

    encoderModel=self.encoderCreate(encoderOld)
    converter=self.converter()
    self.encoder = tf.keras.Model(encoderModel[0],encoderModel[1])
    self.converter= tf.keras.Model(converter[0],converter[1])
    self.decoder = decoderOld


  def encoderCreate(self,frozeEncoder):
     float_to_int_layer = tf.keras.layers.Lambda(lambda x: tf.cast(x*255, dtype=tf.uint8),name="floatConversion1")(frozeEncoder.output[0])
     float_to_int_layer2 = tf.keras.layers.Lambda(lambda x: tf.cast(x*255, dtype=tf.uint8),name="floatConversion2")(frozeEncoder.output[1])
     return frozeEncoder.input,[float_to_int_layer,float_to_int_layer2]
  
 

  def converter(self):
     size=int(100/self.mode)
     newInputRGB=tf.keras.layers.Input(shape=(size,size,3),name="newInputRGB")
     newInputBN=tf.keras.layers.Input(shape=(50,50,1),name="newInputBN")
     int_tofloatRGB = tf.keras.layers.Lambda(lambda x: tf.cast(x, dtype=tf.float32)/255,name="floatREConversion1")(newInputRGB)
     int_tofloatBN = tf.keras.layers.Lambda(lambda x: tf.cast(x, dtype=tf.float32)/255,name="floatREConversion2")(newInputBN)
     return [newInputBN,newInputRGB],[int_tofloatBN,int_tofloatRGB]
  def call(self, x):
    encoded = self.encoder(x)
    recoded=self.converter(encoded)
    decoded = self.decoder(recoded)

    return decoded
  
  def get_config(self):
        config = super(AutoencoderFinal, self).get_config()  # Get parent class config
        config.update({
            'mode': self.mode, 
            'config':self.converter
              # Store any custom attributes
        })
        return config

    # This method defines how to load the model configuration.
  @classmethod
  def from_config(cls, config):
        mode = config['mode']
        # You'd need the oldAutoencoder to recreate the model
        # Assuming you have a method to get an instance of `oldAutoencoder`
        oldAutoencoder = Autoencoder(2)  # You need to define this
        return cls(oldAutoencoder, mode)
  



class AutoencoderRGB(tf.keras.models.Model):
  
  def __init__(self):
    super(AutoencoderRGB, self).__init__()

  
    encoderModel=self.encoderCreate()
    decoderModel=self.decoderCreate()
    self.encoder = tf.keras.Model(encoderModel[0],encoderModel[1])
    self.decoder = tf.keras.Model(decoderModel[0],decoderModel[1])
  
  def encoderCreate(self):
    inputLayer=tf.keras.layers.Input(shape=(100, 100, 3))
    conv1=tf.keras.layers.Conv2D(15, (5,5), activation='relu', padding='same', strides=1)(inputLayer)
    conv2=tf.keras.layers.Conv2D(15, (3,3), activation='relu', padding='same', strides=2)(conv1)
    conv3=tf.keras.layers.Conv2D(10, (3,3), activation='relu', padding='same', strides=1)(conv2)
    output=tf.keras.layers.Conv2D(3,(3,3),activation='sigmoid', padding='same', strides=1)(conv3)
    return inputLayer,output
 
  def decoderCreate(self):
    inputLayer=tf.keras.layers.Input(shape=(50, 50, 3))
    l1=tf.keras.layers.Conv2D(15,(1,1),activation='relu', padding='same', strides=1)(inputLayer)
    upscale= tf.keras.layers.Conv2DTranspose(15, (3,3), activation='relu', padding='same', strides=2)(l1)
    conv1=tf.keras.layers.Conv2D(10, (3,3), activation='relu', padding='same', strides=1)(upscale)
    output=tf.keras.layers.Conv2D(3,(3,3),activation='sigmoid', padding='same', strides=1)(conv1)
    return  inputLayer,output
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)

    return decoded


class AutoencoderFinalRGB(tf.keras.Model):
    def __init__(self, oldAutoencoder):
        super(AutoencoderFinalRGB, self).__init__()

        encoderOld = oldAutoencoder.encoder
        decoderOld = oldAutoencoder.decoder
        for layer in encoderOld.layers:
            layer.trainable = False
        self.encoder = self.encoderCreate(encoderOld)
        self.converter = self.converterCreate()
        self.decoder = decoderOld

    def encoderCreate(self, frozeEncoder):
        encoder_input = frozeEncoder.input  # Shape: (None, 50, 50, 3)
        float_to_int_layer = tf.keras.layers.Lambda(
            lambda x: tf.cast(x * 255, dtype=tf.uint8), name="floatConversion1"
        )(frozeEncoder.output)
        return tf.keras.Model(encoder_input, float_to_int_layer, name="new_encoder")

    def converterCreate(self):
        new_input_rgb = tf.keras.layers.Input(shape=(50, 50, 3), name="newInputRGB")
        int_to_float_layer = tf.keras.layers.Lambda(
            lambda x: tf.cast(x, dtype=tf.float32) / 255, name="floatREConversion1"
        )(new_input_rgb)
        return tf.keras.Model(new_input_rgb, int_to_float_layer, name="converter")

    def call(self, x):
        encoded = self.encoder(x)

        recoded = self.converter(encoded)

        decoded = self.decoder(recoded)

        return decoded
    
class PCA(tf.keras.models.Model):
  
  def __init__(self):
    super(PCA, self).__init__()

  
    encoderModel=self.encoderCreate()
    decoderModel=self.decoderCreate()
    self.encoder = tf.keras.Model(encoderModel[0],encoderModel[1])
    self.decoder = tf.keras.Model(decoderModel[0],decoderModel[1])
  
  def encoderCreate(self):
    inputLayer=tf.keras.layers.Input(shape=(100, 100, 1))
    conv=tf.keras.layers.Conv2D(5,(3,3),activation='relu', padding='same', strides=2)(inputLayer)
    conv2=tf.keras.layers.Conv2D(2,(3,3),activation='relu', padding='same', strides=1)(conv)
    flattened = tf.keras.layers.Flatten()(conv2)
    dense=tf.keras.layers.Dense(50*50,activation="sigmoid")(flattened)
    return inputLayer,dense
 
  def decoderCreate(self):
    inputLayer=tf.keras.layers.Input(shape=(50*50))
    dense=tf.keras.layers.Dense(50*50,activation="sigmoid")(inputLayer)
    reshaped = tf.keras.layers.Reshape((50,50,1))(dense)
    upscale= tf.keras.layers.Conv2DTranspose(2, (3,3), activation='relu', padding='same', strides=2)(reshaped)
    conv1=tf.keras.layers.Conv2D(5,(3,3),activation='relu', padding='same', strides=1)(upscale)
    conv2=tf.keras.layers.Conv2D(1,(3,3),activation='sigmoid', padding='same', strides=1)(conv1)
    return  inputLayer,conv2
  
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)

    return decoded