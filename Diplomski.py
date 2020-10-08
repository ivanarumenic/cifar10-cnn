 -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:30:04 2019

@author: Ivana
"""
### svi importi biblioteka
import keras
import tensorflow as tf
import google
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu
### load dataset iz kerasih datasetova


print('Training data shape : ', x_train.shape, y_train.shape)
print('Testing data shape : ', x_test.shape, y_test.shape)

np.max(x_test[1])
classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)
## prikazi prvu sliku iz train dataset
plt.subplot(121)
plt.imshow(x_train[0,:,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(y_train[0]))

# prikazati prvu sliku iz test dataset
plt.subplot(122)
plt.imshow(x_test[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(y_test[0]))



y_train_one_hot = keras.utils.to_categorical(y_train)
y_test_one_hot = keras.utils.to_categorical(y_test)

# Display the change for category label using one-hot encoding
print('Original label:', y_train[0])
print('After conversion to one-hot:', y_train_one_hot[0])
 from sklearn.model_selection import train_test_split

x_train,x_valid, train_label, valid_label = train_test_split(x_train,y_train_one_hot,test_size = 0.2,random_state=2018)

x_train.shape,x_valid.shape
train_label.shape,valid_label.shape

def plot_random(dataset):
    cifars_random = [300, 2250, 3650, 4000]

# Fill out the subplots with the random images and add shape, min and max values
    for i in range(len(cifars_random)):
        plt.subplot(1, 4, i+1)
        plt.axis('off')
        plt.imshow(dataset[cifars_random[i]],cmap="gray")
        plt.subplots_adjust(wspace=0.5)
        plt.show()
        print("shape: {0}, min: {1}, max: {2}".format(dataset[cifars_random[i]].shape, 
                                                  dataset[cifars_random[i]].min(), 
                                                  dataset[cifars_random[i]].max()))
plot_random(x_test)
from skimage.color import rgb2gray
import skimage.color
### pip install scikit-image --user

x_train= rgb2gray(x_train)
x_valid=rgb2gray(x_valid)
x_test= rgb2gray(x_test) 

x_train = x_train.reshape(-1,32,32,1)
x_test = x_test.reshape(-1,32,32,1)
x_valid = x_valid.reshape(-1,32,32,1)

y_test_one_hot = y_test_one_hot.astype('int32')
train_label = train_label.astype('int32')
valid_label =valid_label.astype('int32')

epoch = 20
learning_rate = 0.001
batch_size = 128
#inputs
n_input = 32



def conv_net(x, weights, biases):  

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 16*16 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 8*8 matrix.
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3, k=2)


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
    
pred = conv_net(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

#calculate accuracy across all the given images and average them out. 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) 
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(n_epochs):
        for batch in range(len(x_train)//batch_size):
            batch_x = x_train[batch*batch_size:min((batch+1)*batch_size,len(x_train))]
            batch_y = y_train[batch*batch_size:min((batch+1)*batch_size,len(y_train))]    
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x,y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y})
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: x_test,y : y_test})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))
    summary_writer.close()

"""Data preprocessing and loadin """

# URL for the data-set on the internet.
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

# Width and height of each image.
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 10

########################################################################
# Various constants used to allocate arrays of the correct size.

# Number of files for the training-set.
_num_files_train = 5

# Number of images for each batch-file in the training-set.
_images_per_file = 10000

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = _num_files_train * _images_per_file

########################################################################
# Private functions for downloading, unpacking and loading data-files.
def download_and_extract(url=data_url, download_dir=hparams.data_dir):
    # Filename for saving the file downloaded from the internet.
    # Use the filename from the URL and add it to the download_dir.
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

    # Check if the file already exists.
    # If it exists then we assume it has also been extracted,
    # otherwise we need to download and extract it now.
    if not os.path.exists(file_path):
        # Check if the download directory exists, otherwise create it.
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # Download the file from the internet.
        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=file_path)

        print()
        print("Download finished. Extracting files.")

        if file_path.endswith((".tar.gz", ".tgz")):
            # Unpack the tar-ball.
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

        print("Done.")


def _get_file_path(filename=""):
    """
    Return the full path of a data-file for the data-set.

    If filename=="" then return the directory of the files.
    """

    return os.path.join(hparams.data_dir, "cifar-10-batches-py/", filename)


def _unpickle(filename):
    """
    Unpickle the given file and return the data.

    Note that the appropriate dir-name is prepended the filename.
    """

    # Create full path for the file.
    file_path = _get_file_path(filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        data = pickle.load(file,encoding='bytes')

    return data

def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images
    
def _load_data(filename):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """

    # Load the pickled data-file.
    data = _unpickle(filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels'])

    # Convert the images.
    images = _convert_images(raw_images)

    return images, cls


def load_class_names():
    # Load the class-names from the pickled file.
    raw = _unpickle(filename="batches.meta")[b'label_names']

    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]

    return names
def load_training_data():
    """
    Load all the training-data for the CIFAR-10 data-set.

    The data-set is split into 5 data-files which are merged here.

    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    return images, cls


def load_testing_data():


    images, cls = _load_data(filename="test_batch")

    images = images[:, :, :, :]
    cls = cls[:]

    return images, cls
download_and_extract(data_url,hparams.data_dir)
_load_data("python.tar.gz")
### load arrays of data###
x_train, y_train = load_training_data()
x_test, y_test = load_testing_data()



"""
def conv_net1(x,rate):
    conv1_weights = tf.Variable(tf.initializers.truncated_normal( mean=0, stddev=0.05).__call__(shape=[5, 5, 3, 64]))
    conv2_weights = tf.Variable(tf.initializers.truncated_normal(mean=0, stddev=0.05).__call__(shape=[5, 5, 64,128]))
    conv3_weights = tf.Variable(tf.initializers.truncated_normal(mean=0, stddev=0.05).__call__(shape=[7,7, 128, 256]))
    conv4_weights = tf.Variable(tf.initializers.truncated_normal(mean=0, stddev=0.05).__call__(shape=[7,7, 256, 512]))
    conv1_bias = tf.Variable(tf.initializers.truncated_normal().__call__(shape=[64]))
    conv2_bias = tf.Variable(tf.initializers.truncated_normal().__call__(shape=[128]))
    conv3_bias = tf.Variable(tf.initializers.truncated_normal().__call__(shape=[256]))
    conv4_bias = tf.Variable(tf.initializers.truncated_normal().__call__(shape=[512]))
    with tf.name_scope('conv1') as scope:
        conv1 = tf.nn.conv2d(x, conv1_weights, strides=[1,1,1,1], padding='SAME')
        conv1 = tf.nn.bias_add(value = conv1,bias = conv1_bias)
        conv1 = tf.nn.relu(conv1)
        conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.name_scope('conv2') as scope:
        conv2 = tf.nn.conv2d(conv1_pool, conv2_weights, strides=[1,1,1,1], padding='SAME')
        conv2 = tf.nn.bias_add(value = conv2,bias = conv2_bias)
        conv2_bn = tf.layers.batch_normalization(conv2)
        conv2_bn = tf.nn.relu(conv2_bn)
        conv2_pool = tf.nn.max_pool(conv2_bn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    
    
    with tf.name_scope('conv3') as scope:
        conv3 = tf.nn.conv2d(conv2_pool, conv3_weights, strides=[1,1,1,1], padding='SAME') + conv3_bias
        conv3_bn = tf.layers.batch_normalization(conv3)       
        conv3_bn = tf.nn.relu(conv3_bn)
        conv3_pool = tf.nn.max_pool(conv3_bn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') 
        conv3_do = tf.layers.dropout(conv3_pool,rate = rate)
    with tf.name_scope('conv4') as scope:
        conv4 = tf.nn.conv2d(conv3_do, conv4_weights, strides=[1,1,1,1], padding='SAME') + conv4_bias
        conv4_bn = tf.layers.batch_normalization(conv4)       
        conv4_bn = tf.nn.relu(conv4_bn)
        conv4_pool = tf.nn.max_pool(conv4_bn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  
        conv4_do = tf.layers.dropout(conv4_pool,rate = rate)
    with tf.name_scope('flatten') as scope:
        flat = tf.contrib.layers.flatten(conv4_pool)  
    
    with tf.name_scope('fully_connected1') as scope:
        full = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
        full = tf.layers.batch_normalization(full)
        full = tf.layers.dropout(full, rate =rate)
    with tf.name_scope('fully_connected2') as scope:
        full1 = tf.contrib.layers.fully_connected(inputs=full, num_outputs=256, activation_fn=tf.nn.relu)
        full1= tf.layers.batch_normalization(full1)
        full1 = tf.layers.dropout(full1,rate = rate)

    with tf.name_scope('out') as scope:
        out = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=10, activation_fn=None)
    
    return out

def conv_net2(x,rate):
    conv1_weights = tf.Variable(tf.initializers.truncated_normal( mean=0, stddev=0.05).__call__(shape=[5, 5, 3, 64]))
    conv2_weights = tf.Variable(tf.initializers.truncated_normal(mean=0, stddev=0.05).__call__(shape=[5, 5, 64,128]))
    conv3_weights = tf.Variable(tf.initializers.truncated_normal(mean=0, stddev=0.05).__call__(shape=[7,7, 128, 256]))
    conv4_weights = tf.Variable(tf.initializers.truncated_normal(mean=0, stddev=0.05).__call__(shape=[7,7, 256, 512]))
    conv1_bias = tf.Variable(tf.initializers.truncated_normal().__call__(shape=[64]))
    conv2_bias = tf.Variable(tf.initializers.truncated_normal().__call__(shape=[128]))
    conv3_bias = tf.Variable(tf.initializers.truncated_normal().__call__(shape=[256]))
    conv4_bias = tf.Variable(tf.initializers.truncated_normal().__call__(shape=[512]))
    with tf.name_scope('conv1') as scope:
        conv1 = tf.nn.conv2d(x, conv1_weights, strides=[1,1,1,1], padding='SAME')
        conv1 = tf.nn.bias_add(value = conv1,bias = conv1_bias)
        conv1 = tf.nn.relu(conv1)
        conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.name_scope('conv2') as scope:
        conv2 = tf.nn.conv2d(conv1_pool, conv2_weights, strides=[1,1,1,1], padding='SAME')
        conv2 = tf.nn.bias_add(value = conv2,bias = conv2_bias)
        conv2_bn = tf.layers.batch_normalization(conv2)
        conv2_bn = tf.nn.relu(conv2_bn)
        conv2_pool = tf.nn.max_pool(conv2_bn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    
    
    with tf.name_scope('conv3') as scope:
        conv3 = tf.nn.conv2d(conv2_pool, conv3_weights, strides=[1,1,1,1], padding='SAME') + conv3_bias
        conv3_bn = tf.layers.batch_normalization(conv3)       
        conv3_bn = tf.nn.relu(conv3_bn)
        conv3_pool = tf.nn.max_pool(conv3_bn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') 
        conv3_do = tf.layers.dropout(conv3_pool,rate = rate)
    with tf.name_scope('conv4') as scope:
        conv4 = tf.nn.conv2d(conv3_do, conv4_weights, strides=[1,1,1,1], padding='SAME') + conv4_bias
        conv4_bn = tf.layers.batch_normalization(conv4)       
        conv4_bn = tf.nn.relu(conv4_bn)
        conv4_pool = tf.nn.max_pool(conv4_bn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  
        conv4_do = tf.layers.dropout(conv4_pool,rate = rate)
    with tf.name_scope('flatten') as scope:
        flat = tf.contrib.layers.flatten(conv4_do)  
    
    with tf.name_scope('fully_connected1') as scope:
        full = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=512, activation_fn=tf.nn.relu)
        full = tf.layers.batch_normalization(full)
        full = tf.layers.dropout(full, rate =rate)
    with tf.name_scope('fully_connected2') as scope:
        full1 = tf.contrib.layers.fully_connected(inputs=full, num_outputs=1024, activation_fn=tf.nn.relu)
        full1= tf.layers.batch_normalization(full1)
        full1 = tf.layers.dropout(full1,rate = rate)

    with tf.name_scope('out') as scope:
        out = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=10, activation_fn=None)
    
    return out
def conv_net3(x,rate):
    conv1_weights = tf.Variable(tf.initializers.truncated_normal( mean=0, stddev=0.05).__call__(shape=[5, 5, 3, 64]))
    conv2_weights = tf.Variable(tf.initializers.truncated_normal(mean=0, stddev=0.05).__call__(shape=[5, 5, 64,128]))
    conv3_weights = tf.Variable(tf.initializers.truncated_normal(mean=0, stddev=0.05).__call__(shape=[7,7, 128, 256]))
    conv4_weights = tf.Variable(tf.initializers.truncated_normal(mean=0, stddev=0.05).__call__(shape=[7,7, 256, 512]))
    
    conv1_bias = tf.Variable(tf.initializers.truncated_normal().__call__(shape=[64]))
    conv2_bias = tf.Variable(tf.initializers.truncated_normal().__call__(shape=[128]))
    conv3_bias = tf.Variable(tf.initializers.truncated_normal().__call__(shape=[256]))
    conv4_bias = tf.Variable(tf.initializers.truncated_normal().__call__(shape=[512]))
    with tf.name_scope('conv1') as scope:
        conv1 = tf.nn.conv2d(x, conv1_weights, strides=[1,1,1,1], padding='SAME')
        conv1 = tf.nn.bias_add(value = conv1,bias = conv1_bias)
        conv1 = tf.nn.relu(conv1)
        conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.name_scope('conv2') as scope:
        conv2 = tf.nn.conv2d(conv1_pool, conv2_weights, strides=[1,1,1,1], padding='SAME')
        conv2 = tf.nn.bias_add(value = conv2,bias = conv2_bias)
        conv2_bn = tf.layers.batch_normalization(conv2)
        conv2_bn = tf.nn.relu(conv2_bn)
        conv2_pool = tf.nn.max_pool(conv2_bn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    
        conv2_pool = tf.layers.dropout(conv2_pool,rate=rate)
    with tf.name_scope('conv3') as scope:
        conv3 = tf.nn.conv2d(conv2_pool, conv3_weights, strides=[1,1,1,1], padding='SAME') + conv3_bias
        conv3_bn = tf.layers.batch_normalization(conv3)       
        conv3_bn = tf.nn.relu(conv3_bn)
        conv3_pool = tf.nn.max_pool(conv3_bn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') 
        conv3_do = tf.layers.dropout(conv3_pool,rate = rate)
    with tf.name_scope('conv4') as scope:
        conv4 = tf.nn.conv2d(conv3_do, conv4_weights, strides=[1,1,1,1], padding='SAME') + conv4_bias
        conv4_bn = tf.layers.batch_normalization(conv4)       
        conv4_bn = tf.nn.relu(conv4_bn)
        conv4_pool = tf.nn.max_pool(conv4_bn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  
        conv4_do = tf.layers.dropout(conv4_pool,rate = rate)
    with tf.name_scope('flatten') as scope:
        flat = tf.contrib.layers.flatten(conv4_do)  
    
    with tf.name_scope('fully_connected1') as scope:
        full = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=512, activation_fn=tf.nn.relu)
        full = tf.layers.batch_normalization(full)
        full = tf.layers.dropout(full,rate = rate)
    with tf.name_scope('fully_connected2') as scope:
        full1 = tf.contrib.layers.fully_connected(inputs=full, num_outputs=1024, activation_fn=tf.nn.relu)
        full1= tf.layers.batch_normalization(full1)
        full1 = tf.layers.dropout(full1,rate = rate)

    with tf.name_scope('out') as scope:
        out = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=10, activation_fn=None)
    
    return out
def conv_net4(x,rate):
    conv1_weights = tf.Variable(tf.initializers.truncated_normal( mean=0, stddev=0.05).__call__(shape=[5, 5, 3, 64]))
    conv2_weights = tf.Variable(tf.initializers.truncated_normal(mean=0, stddev=0.05).__call__(shape=[5, 5, 64,128]))
    conv3_weights = tf.Variable(tf.initializers.truncated_normal(mean=0, stddev=0.05).__call__(shape=[7,7, 128, 256]))
  #  conv4_weights = tf.Variable(tf.initializers.truncated_normal(mean=0, stddev=0.05).__call__(shape=[7,7, 256, 512]))
    
    conv1_bias = tf.Variable(tf.initializers.truncated_normal().__call__(shape=[64]))
    conv2_bias = tf.Variable(tf.initializers.truncated_normal().__call__(shape=[128]))
    conv3_bias = tf.Variable(tf.initializers.truncated_normal().__call__(shape=[256]))
   # conv4_bias = tf.Variable(tf.initializers.truncated_normal().__call__(shape=[512]))
    with tf.name_scope('conv1') as scope:
        conv1 = tf.nn.conv2d(x, conv1_weights, strides=[1,1,1,1], padding='SAME')
        conv1 = tf.nn.bias_add(value = conv1,bias = conv1_bias)
        conv1 = tf.nn.relu(conv1)
        conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.name_scope('conv2') as scope:
        conv2 = tf.nn.conv2d(conv1_pool, conv2_weights, strides=[1,1,1,1], padding='SAME')
        conv2 = tf.nn.bias_add(value = conv2,bias = conv2_bias)
        conv2_bn = tf.layers.batch_normalization(conv2)
        conv2_bn = tf.nn.relu(conv2_bn)
        conv2_pool = tf.nn.max_pool(conv2_bn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    
        conv2_pool = tf.layers.dropout(conv2_pool,rate=rate)
    with tf.name_scope('conv3') as scope:
        conv3 = tf.nn.conv2d(conv2_pool, conv3_weights, strides=[1,1,1,1], padding='SAME') + conv3_bias
        conv3_bn = tf.layers.batch_normalization(conv3)       
        conv3_bn = tf.nn.relu(conv3_bn)
        conv3_pool = tf.nn.max_pool(conv3_bn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') 
        conv3_do = tf.layers.dropout(conv3_pool,rate = rate)
        """
    """
    with tf.name_scope('conv4') as scope:
        conv4 = tf.nn.conv2d(conv3_do, conv4_weights, strides=[1,1,1,1], padding='SAME') + conv4_bias
        conv4_bn = tf.layers.batch_normalization(conv4)       
        conv4_bn = tf.nn.relu(conv4_bn)
        conv4_pool = tf.nn.max_pool(conv4_bn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  
        conv4_do = tf.layers.dropout(conv4_pool,rate = rate)
    """
"""
    with tf.name_scope('flatten') as scope:
        flat = tf.contrib.layers.flatten(conv3_do)  
    
    with tf.name_scope('fully_connected1') as scope:
        full = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=512, activation_fn=tf.nn.relu)
        full = tf.layers.batch_normalization(full)
        full = tf.layers.dropout(full,rate = rate)
    with tf.name_scope('fully_connected2') as scope:
        full1 = tf.contrib.layers.fully_connected(inputs=full, num_outputs=1024, activation_fn=tf.nn.relu)
        full1= tf.layers.batch_normalization(full1)
        full1 = tf.layers.dropout(full1,rate = rate)

    with tf.name_scope('out') as scope:
        out = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=10, activation_fn=None)
    
    return out
    """