
# coding: utf-8

# In[11]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np

import tensorflow as tf

FLAGS = None


def main(_):
    
    # importing training labels and images
    trainingImages = np.load("trainingVectors.npy")
    trainingLabels = np.load("trainingLabels.npy")
    
    # importing validation labels and images
    validationImages = np.load("validationVectors.npy")
    validationLabels = np.load("validationLabels.npy")
    
    # importing testing labels and images
    testingImages = np.load("testingVectors.npy")
    testingLabels = np.load("testingLabels.npy")
    
    # dimesion of images and labels
    dimImage = trainingImages.shape[1]
    dimLabel = trainingLabels.shape[1]
    
    # Hyper parameters
    image_width_px = 28
    image_height_px = 28
    n_epochs = 800
    b_size = 38
    
    # making the placeholders for input and output
    x = tf.placeholder(tf.float32, [None, dimImage])
    y_ = tf.placeholder(tf.float32, [None, dimLabel])

    # Weights and biases of the first convolution layer
    W_conv1 = weight_variable([5, 5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    # Reshaping image matrix before applying convolution 		    
    x_image = tf.reshape(x, [-1, image_width_px, image_height_px, 3, 1])
    
    # applying first layer convolution
    h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)
    # applying first layer pooling
    h_pool1 = max_pool_3d(h_conv1)
    
    # Weights and biases of the first convolution layer
    W_conv2 = weight_variable([5, 5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    # applying second layer convolution
    h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
    # applying second layer pooling
    h_pool2 = max_pool_3d(h_conv2)
    
    # Weights and biases of the first full connected layer
    W_fc1 = weight_variable([3136, 1024])
    b_fc1 = bias_variable([1024])

    # reshaping and pooling first full connected layer	
    h_pool2_flat = tf.reshape(h_pool2, [-1, 3136])
    # applying regression to the first full connected layer
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    # drop out
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # Weights and biases of the second full connected layer
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])

    # final convolution of the second full connected layer
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
   
    # finding out cross entropy until mean is minimized
    cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    
    #training model
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #Initializing session
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      
      # Training epochs starts
      for i in range(n_epochs):
            batch = next_batch(b_size, trainingImages,trainingLabels)
            if i % 5 == 0:
                train_accuracy = accuracy.eval(feed_dict={
              x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
      acc = accuracy.eval(feed_dict={x: validationImages, y_:validationLabels, keep_prob:1.0})	
      print('test accuracy %g' % acc)
      Stxt = (str(len(trainingImages)) + " " + str(image_width_px)+" "+str(image_height_px)+" "+str(n_epochs)+" "+str(b_size)+" "+str(acc) + "\n")
      F = open("Results.txt","a")
      F.write(Stxt)
      F.close()


# defining weight variable 
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#defining bias variable
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# applying 3d convolution for 3 channel images
def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

# applying 3d pooling (max) for 3 channel images
def max_pool_3d(x):
  return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                        strides=[1, 2, 2, 2, 1], padding='SAME')

# finding out the next batch of the iteration
# used during training
def next_batch(batch_size, images, labels):
    vectorSize = (images.shape)[0]
    shuffleIndex = np.random.randint(vectorSize,size=batch_size)
    imagesList = []
    labelsList = []
    for i in shuffleIndex:
        imagesList.append(images[i])
        labelsList.append(labels[i])
    imageNp = np.array(imagesList)
    labelsNp = np.array(labelsList)
    batch = (imageNp,labelsNp)
    return batch

if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])


