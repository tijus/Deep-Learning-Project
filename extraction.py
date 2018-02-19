'''
This programs extract images and their labels and saves
them into numpy file in the form of numpy array
'''

import pandas as pd
import cv2
import numpy as np
from random import randint

labelFile = pd.read_csv('attributes.csv');
outputLabel = 'Eyeglasses'
nameLabel = 'image_name'

labels = labelFile[:][outputLabel]
labels = labels.tolist()
labels1 = [1 if i==0 else 0 for i in labels]

labels = np.array(labels)
labels1 = np.array(labels1)
labels = np.vstack([labels1,labels])
labels = labels.T

paths = labelFile[:][nameLabel]
dataFraction = int(0.05 * len(labels))  # fraction of data to be used -> set to .01 when using on local system (1%)
trainFraction = int(0.8 * dataFraction)
validationFraction = int(0.9 * dataFraction)
imDim1 = 28
imDim2 = 28

def extractLabels(flag,augfract):
    '''
    	extractLabels: This function extract labels from the file
    	Input: 
   		flag: a specifier that determine whether to use augmentation
   		augfract: fraction of the training set used for augmentation
    	Output:
   		returns the testing, training and validation labels and saves 
   		it into their respective files
    '''
    
    # labels for different sets
    trainingLabels = labels[0 : trainFraction]
    augmentedLabels = labels[0 : int(augfract*trainFraction)]
    print (trainingLabels.shape)
    print (augmentedLabels.shape)
    if flag == 1:
        trainingLabels = np.concatenate((trainingLabels, augmentedLabels), axis=0)
    #trainingLabels.to_csv('trainingLabels.csv', sep=',',header='None')
    print (trainingLabels.shape)
    np.save('trainingLabels', trainingLabels)

    validationLabels = labels[trainFraction : validationFraction]
    #validationLabels.to_csv('validationLabels.csv', sep=',',header='None')
    np.save('validationLabels', validationLabels)

    testingLabels = labels[validationFraction : dataFraction]
    #testingLabels.to_csv('testingLabels.csv', sep=',',header='None')
    np.save('testingLabels',testingLabels)


def readImages(flag,augfract):
    '''
   	readImages: This function extract images from the file
   	Input: 
   		flag: a specifier that determine whether to use augmentation
   		augfract: fraction of the training set used for augmentation
   	Output:
   		returns the testing, training and validation images and saves 
   		it into their respective files
    '''
    #relative path append to paths -> '../img/'
    augDist = int(augfract*trainFraction)
    pathsList = paths.tolist()
    imagePaths = ['img/' + s for s in pathsList]
    trainPaths = imagePaths[0 : trainFraction]
    augmentedPaths = imagePaths[0 : augDist]
    validationPaths = imagePaths[trainFraction : validationFraction]
    testingPaths = imagePaths[validationFraction : dataFraction]
    trainVectors = np.empty((0, imDim1 * imDim2 * 3), float)
    for ipath in trainPaths:
        img = cv2.imread(ipath);
        img = cv2.resize(img, (imDim1, imDim2))
        flattenedImg = img.flatten()
        trainVectors = np.vstack([trainVectors, flattenedImg])
    if flag == 1:
            print("Augmentation in progress...")
            transformedVector = imageAugmentation(augmentedPaths,augDist)
            trainVectors = np.vstack([trainVectors, transformedVector])
    #print(trainVectors.shape)
    np.save('trainingVectors', trainVectors)
    
    #np.savetxt('trainingVectors.csv', trainVectors, delimiter = ',')

    validationVectors = np.empty((0, imDim1 * imDim2 * 3), float);
    for ipath in validationPaths:
        img = cv2.imread(ipath);
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (imDim1, imDim2))
        flattenedImg = img.flatten()
        validationVectors = np.vstack([validationVectors, flattenedImg]);
        print(ipath)

    np.save('validationVectors', validationVectors)
    #np.savetxt('validationVectors.csv', validationVectors, delimiter = ',')

    testingVectors = np.empty((0, imDim1 * imDim2 * 3), float);
    for ipath in testingPaths:
        img = cv2.imread(ipath);
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (imDim1, imDim2))
        flattenedImg = img.flatten()
        testingVectors = np.vstack([testingVectors, flattenedImg]);
        print(ipath)

    np.save('testingVectors', testingVectors)
    #np.savetxt('testingVectors.csv', testingVectors, delimiter=',')
    
def imageAugmentation(augmentedPath, n_augmented):
    '''
   	imageAugmentation: This function performs image augmentation
   			i.e. rotation, translation and crop to find
   			new sets of data
  	 Input: 
   		augmentedPath: image path of the augmented set
   		n_augmented: number of images to be augmented
   	Output:
   		returns a numpy array of augmented image
    '''
    translate = 0.5*n_augmented
    rotate = 0.8*n_augmented
    crop = 1.0*n_augmented
    transformedVectors = np.empty((0, imDim1 * imDim2 * 3), float)
    for i in range(n_augmented):
        img = cv2.imread(augmentedPath[i]);
        if i<translate:
            #translation
            M = np.float32([[1,0,10],[0,1,5]])
            transformedImage = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
            #rotation
        elif i<rotate and i>=translate:
            R = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),45,1)
            transformedImage = cv2.warpAffine(img,R,(img.shape[1],img.shape[0]))
            #testing
        elif i<crop and i>rotate:
            crop_Px = randint(0,5)
            transformedImage = img[crop_Px:img.shape[0]-crop_Px,crop_Px:img.shape[1]-crop_Px]
        transformedImage = cv2.resize(transformedImage, (imDim1, imDim2))
        flattenedImg = transformedImage.flatten()
        transformedVectors = np.vstack([transformedVectors, flattenedImg])
    return transformedVectors

def main():
    # for training set without augmentation
    extractLabels(0,0)
    # for training set with augmentation    
    #extractLabels(1,0.25);
    
    # fot training labels without augmentation
    readImages(0,0);
    # for training labels with augmentation
    #readImages(1,0.25);

main()

