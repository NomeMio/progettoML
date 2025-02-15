import csv
import io
from multiprocessing.context import BaseContext
from os import splice
from os import listdir
from PIL import Image# load all images in a directory
from matplotlib.pyplot import flag


def checkSizeOfset(pathToDir):
        mapSize={}
        for filename in listdir(pathToDir):
                im = Image.open(pathToDir + filename)
                width, height = im.size
                if (width,height) not in mapSize:
                        mapSize[(width,height)]=1
                else:
                        mapSize[(width,height)]+=1
        
        sizeOfImages=0
        minW=999999
        minH=999999
        maxW=0
        maxH=0
        for tuple in mapSize:
                if tuple[0]<minW :
                         minW=tuple[0]
                if tuple[1]<minH :
                          minH=tuple[1]
                if tuple[0]>maxW :
                        maxW=tuple[0]
                if tuple[1]>maxH :
                        maxH=tuple[1]
                sizeOfImages+=tuple[0]*tuple[1]*3*mapSize[tuple]
        sizeOfImages=round(sizeOfImages/pow(10,9),2)
        return (sizeOfImages,(minW,minH),(maxW,maxH))

import os
def resizeImages(folderPath,newPrefix,size,overwrite=False):
        newFolder=newPrefix
        try:
                os.mkdir(newFolder)
        except FileExistsError:
                if overwrite==True:
                        for file in listdir(newFolder):
                                file.remove()
        for filename in listdir(folderPath):
                im = Image.open(folderPath +"/"+ filename)
                newName=str.split(filename,'.')
                newName=newName[0]+'.'+newName[1]#useless code that i used before to give new prefixes and sufixes
                newImage=im.resize(size)
                newImage.save(newFolder+'/'+newName)
        print('resized images to size',size)
        return newFolder+'/'


def printData(implicitValues,path):
        print(path)
        print(implicitValues[0]," GB of images")
        print('min dimension:',implicitValues[1])
        print('max dimension:',implicitValues[2])
        

from tensorflow import keras


import random
import numpy as np

import tensorflow as tf
class getSamples:
        def __init__(self,directory,imgSize,batchSize=100,seed=1,split=0.2,training=True):
                mode="training" if training else "validation" 
                train_ds = keras.utils.image_dataset_from_directory(
                        directory=directory,
                        validation_split=split if split!=0 else None,
                        label_mode=None,
                        subset=mode if split!=0 else None,
                        seed=seed,
                        batch_size=batchSize)
                self.imageSize=imgSize
                self.dir=directory
                self.batchSize=batchSize
                self.filesDirs=train_ds.file_paths
                self.len=len(self.filesDirs)
                self.read=0
                for index,name in enumerate(self.filesDirs):
                        self.filesDirs[index]=str.replace(name,directory+"/","")
                random.shuffle(self.filesDirs)
        def getBatch(self,reshae=True):
                if self.read==self.len:
                        return np.zeros((1,1))
                temp=self.read+self.batchSize
                temp=temp if temp<self.len else self.len 
                array=[]
                if reshae :
                        array=np.zeros((temp,self.imageSize[0]*self.imageSize[1]*3))
                else:
                        array=np.zeros((temp,self.imageSize[0],self.imageSize[1],3))
                for i in range(self.read,temp):
                        im = Image.open(self.dir+"/"+self.filesDirs[i])
                        im = np.array(im)
                        if reshae :im=im.reshape(1,-1)
                        im =im/ 255
                        array[i]=im[..., :3]
                self.read=temp
                print(self.len,' ',self.read)
                print('got another batch, ',self.len-self.read,' remaning',' we are using ',temp)
                return array
        def getRandomImage(self):
                i=random.randint(0,self.len-1)
                image=Image.open(self.dir+"/"+self.filesDirs[i])
                image=np.array(image)
                image=image.reshape(1,-1)
                image=image/255
                return  image

        def returnPaths(self):
                return self.filesDirs,self.dir


        def restPool(self):
                random.shuffle(self.filesDirs)
                self.read=0
                        
        def sizeImage(self):
                return self.imageSize[0]*self.imageSize[1]*3
                


from tensorflow import image as imageTF
from tensorflow import constant as constantTF 
def getStats(encoded,shape,normal):
    media=0
    for i in range(0,len(encoded)):
            dec= encoded[i]
            nor= normal[i]
            for j in range(0,shape[2]):
                 media+=0.333*imageTF.ssim(constantTF(np.reshape(dec[:,:,j],(shape[0],shape[1],1)), dtype='float32'), constantTF(np.reshape(nor[:,:,j],(shape[0],shape[1],1)), dtype='float32'), max_val=1.0).numpy()
    ssimMedia=round(media/len(encoded),3)
    return [ssimMedia]

from tensorflow import expand_dims as expand_dimsTF
from skimage.metrics import structural_similarity as ssimSKI

def measure_jpeg_compression_to_csv(image_paths,size ,qualities, subsampling_modes, output_csv):
    results = []
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        image = Image.open(image_path)
        # Convert RGBA to RGB if necessary
        if image.mode == "RGBA":
            image = image.convert("RGB")
        process_image = lambda x:expand_dimsTF( np.array(x)/255,axis=0)
        for quality in qualities:
            for subsampling in subsampling_modes:
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=quality, subsampling=subsampling)
                imageCompressed=Image.open(buffer)
                dec= np.array(image)/255
                nor= np.array(imageCompressed)/255
                media=0
                for j in range(0,3):
                        media+=0.333*tf.image.ssim(tf.constant(np.reshape(dec[:,:,j],(100,100,1)), dtype='float32'), tf.constant(np.reshape(nor[:,:,j],(100,100,1)), dtype='float32'), max_val=1.0).numpy()
        
                file_size = buffer.tell()
                bit_rate=file_size/(size[0]*size[1]*size[2])
                bit_rate=round(bit_rate,5)
                buffer.close()
                results.append({
                    "image_name": image_name,
                    "name": str(quality)+" subsampling "+subsampling,
                    "bit_rate": bit_rate,
                    "ssim":media
                })
    with open(output_csv, mode="w+", newline="") as csv_file:
        fieldnames = ["image_name", "name", "bit_rate","ssim"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    print(f"Results saved to {output_csv}")


import matplotlib.pyplot as plt
def printSamples(autoencoder,batch,nToPlot,latentSizes):
 
 n=5
 startI=0
 a,b = autoencoder.encoder(batch)
 a=a.numpy()
 b=b.numpy()
 decoded_imgs = autoencoder.decoder(autoencoder.converter([a,b])).numpy()
 plt.figure(figsize=(20, 20))
 startI=0
 n=nToPlot
 for i in range(startI,n):
   # display original
   ax = plt.subplot(4, 5, i-startI + 1)
   plt.imshow(batch[i])
   plt.title("original")
   plt.gray()
   ax.get_xaxis().set_visible(False)
   ax.get_yaxis().set_visible(False)
 
   ax = plt.subplot(4, 5, i-startI + 1 +5)
   plt.imshow(a[i].reshape(latentSizes[0]),cmap='gray', vmin=0, vmax=255)
   plt.title("latent")
   plt.gray()
   ax.get_xaxis().set_visible(False)
   ax.get_yaxis().set_visible(False)
   
   ax = plt.subplot(4, 5, i-startI + 1 +2*5)
   plt.imshow(b[i].reshape(latentSizes[1]))
   plt.title("latent")
   plt.gray()
   ax.get_xaxis().set_visible(False)
   ax.get_yaxis().set_visible(False)
   # display reconstruction
   ax = plt.subplot(4, 5,i-startI + 1 + 3*5)
   plt.imshow(decoded_imgs[i])
   plt.title("reconstructed")
   plt.gray()
   ax.get_xaxis().set_visible(False)
   ax.get_yaxis().set_visible(False)
 plt.show()


def rateGraph(data):
        pivot_table = data.pivot_table(index="image_name", columns="name", values="bit_rate")
        plt.figure(figsize=(10, 6))
        for quality in pivot_table.columns:
            plt.plot(pivot_table.index, pivot_table[quality], label=f"Jpeg {quality}")
        plt.xlabel("")
        plt.ylabel("Byte Rate")
        plt.title("Byte Rate for all test images")
        plt.legend(title="Names")
        plt.grid(False)
        plt.xticks(ticks=range(len(pivot_table.index)), labels=range(len(pivot_table.index)))
        plt.tight_layout()
        plt.show()
def ssimGraph(data):
        group_size = 1  
        data["group_index"] = data.groupby("name").cumcount() // group_size

        grouped_data = data.groupby(["group_index", "name"])["ssim"].mean().reset_index()

        pivot_table = grouped_data.pivot(index="group_index", columns="name", values="ssim")
        plt.figure(figsize=(10, 6))
        for quality in pivot_table.columns:
            plt.plot(pivot_table.index, pivot_table[quality], label=f"{quality}")

        plt.xlabel("")
        plt.ylabel("ssim")
        plt.title("ssim for test images")
        plt.legend(title="names")
        plt.grid(False)

        plt.xticks(ticks=range(len(pivot_table.index)), labels=range(len(pivot_table.index)))

        plt.tight_layout()
        plt.show()

def printScatter(data):
        data['bit_rate'] = data['bit_rate'] 
        grouped_data = data.groupby('name')[['ssim', 'bit_rate']].mean()
        plt.figure(figsize=(10, 6))
        for name in grouped_data.index:
            plt.scatter(grouped_data.loc[name, 'bit_rate'], 
                        grouped_data.loc[name, 'ssim'], 
                        label=name, 
                        alpha=0.7, 
                        s=100) 
        plt.title('SSIM vs. Bit Rate by Name', fontsize=16)
        plt.xlabel('Byte Rate (Mean)', fontsize=14)
        plt.ylabel('SSIM (Mean)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Name', fontsize=12)
        plt.tight_layout()
        plt.show()


def prepareImagesOfAllModels(models,batch):
       images=0
       for ind,mode in enumerate(models):
                result=mode.decoder(mode.converter(mode.encoder(batch))).numpy()
                process_array = lambda x: (np.clip(x,0.0,1.0) * 255).astype(np.uint8)
                img=process_array(result)
                if(ind==0):
                       images=np.expand_dims(img, axis=0)
                else:
                       images=np.append(images,np.expand_dims(img, axis=0),axis=0)
       return images

def printAllDec(originalImg,labelArray,images):
 quals=[10,50,70,90] 
 process_array = lambda x: (np.clip(x,0.0,1.0) * 255).astype(np.uint8)
 n=len(originalImg)

 plt.figure(figsize=(20, 40))
 for i in range(0,len(originalImg)):
   ax = plt.subplot(8, 5, i + 1)
   plt.imshow(originalImg[i])
   plt.title("original")
   plt.gray()
   ax.get_xaxis().set_visible(False)
   ax.get_yaxis().set_visible(False)
 
   for indx,label in enumerate(labelArray):
        ax = plt.subplot(8, 5, i + 1+n*(indx+1))
        plt.imshow(images[indx][i])
        plt.title(label)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)     
   lenM=len(labelArray)     
   for indx,q in enumerate(quals):
        ax = plt.subplot(8, 5, i +lenM*n+ 1+n*(indx+1))
        buffer = io.BytesIO()
        image=Image.fromarray(process_array(originalImg[i]))
        image.save(buffer, format="JPEG", quality=q)
        imageCompressed=Image.open(buffer)
        dec= np.array(imageCompressed)
        plt.imshow(dec)
        plt.title("jpeg "+str(q))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
 plt.show()