# To increase the diversity by applying transformations or modifications to the original dataset.
# Import libraries.
import albumentations as A
import cv2
import os
import time
import multiprocessing

# Define the path to the root directory of dataset.
dir="/home/shabeen/data/103classes_plants/"
classes=os.listdir(dir)
classes=sorted([item  for item in classes if '.' not in item and item!='AllDatasets'])
print(len(classes),classes)

# Specify the number of images per class to be needed.
limit=600

# data transformation.
transform = A.Compose([
        A.RandomRotate90(p=1),
        A.Flip(p=1),
        #A.Transpose(p=0.5),
        A.GaussNoise(p=0.5),
        A.Sharpen(p=0.5,alpha=(0.2,0.4)),
        A.RandomToneCurve(p=0.5,scale=0.2,per_channel=True),
        A.ToGray(p=0.1),
        A.OneOf([
            A.MotionBlur(p=0.6),
            A.MedianBlur(blur_limit=3, p=0.6),
            A.Blur(blur_limit=3, p=0.6),], p=0.5),
        #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
        A.OneOf([
            A.OpticalDistortion(p=0.7),
            A.GridDistortion(p=0.7),], p=0.5),
        A.OneOf([
            A.CLAHE(clip_limit=2,p=0.6),
            A.RandomBrightnessContrast(p=0.7),], p=0.5),
        A.HueSaturationValue(p=0.4),
    ])


# Function for creating new image sample.
def augment(cls):
    print("ID of process running worker: {}".format(os.getpid()))
	
    def repeat(nmbr,count):
        for n,i in zip(range(count,count+nmbr),images[:nmbr]):
            image = cv2.imread(os.path.join(src,i))
            h,w,_=image.shape
            augmented_image =A.CenterCrop(height=int(h*0.9),width=int(w*0.9),p=1)(image=image)['image']
            augmented_image = transform(image=augmented_image)['image']
            final_img=f"{src}/augmented_{n}.jpg"
            cv2.imwrite(final_img,augmented_image)
            print(final_img)


    src= dir +'/'+ cls
    images=os.listdir(src)
    
    images_n=len(images)
    count=0
    to_add=limit-images_n
    while to_add >0:
    
        if to_add/images_n>1:
            repeat(images_n,count)  
            to_add=to_add-images_n
            count=count+images_n
    
        else:
            repeat(to_add,count)
            to_add=0
    
# Use multiprocessing for efficient data augmentation.
start=time.time()
p = multiprocessing.Pool() 
result = p.map(augment,classes)
end=time.time()
print("timetake",end-start)   