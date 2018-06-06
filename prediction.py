# Starter code for CS 165B HW4

"""
Implement the testing procedure here. 

Inputs:
    Given the folder named "hw4_test" that is put in the same directory of your "predictio.py" file, like:
    - Main folder
        - "prediction.py"
        - folder named "hw4_test" (the exactly same as the uncompressed hw4_test folder in Piazza)
    Your "prediction.py" need to give the following required output.

Outputs:
    A file named "prediction.txt":
        * The prediction file must have 10000 lines because the testing dataset has 10000 testing images.
        * Each line is an integer prediction label (0 - 9) for the corresponding testing image.
        * The prediction results must follow the same order of the names of testing images (0.png – 9999.png).
    Notes: 
        1. The teaching staff will run your "prediction.py" to obtain your "prediction.txt" after the competition ends.
        2. The output "prediction.txt" must be the same as the final version you submitted to the CodaLab, 
        elsewise you will be given 0 score for your hw4.


**!!!!!!!!!!Important Notes!!!!!!!!!!**
    To open the folder "hw4_test" or load other related files, 
    please use open('./necessary.file') instaed of open('some/randomly/local/directory/necessary.file').

    For instance, in the student Jupyter's local computer, he stores the source code like:
    - /Jupyter/Desktop/cs165B/hw4/prediction.py
    - /Jupyter/Desktop/cs165B/hw4/hw4_test
    If he use os.chdir('/Jupyter/Desktop/cs165B/hw4/hw4_test'), this will cause an IO error 
    when the teaching staff run his code under other system environments.
    Instead, he should use os.chdir('./hw4_test').


    If you use your local directory, your code will report an IO error when the teaching staff run your code,
    which will cause 0 socre for your hw4.
"""
import os
import skimage
from skimage import data
from glob import glob
from PIL import Image
from numpy import array
from skimage.color import rgb2gray
from skimage import transform
import tensorflow as tf

def load_training_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        files = [os.path.join(label_directory, f)
                 for f in os.listdir(label_directory)
                 if f.endswith(".png")]
        files.sort()
        for f in files:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    images = [transform.resize(image, (28, 28)) for image in images]
    images = rgb2gray(array(images))
    labels = array(labels)
    
    return images, labels

images, labels = load_training_data("./hw4_train")


x = tf.placeholder(dtype = tf.float32, shape = [None, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])


images_flat = tf.contrib.layers.flatten(x)

logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))
# Define an optimizer 
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.set_random_seed(1234)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(201):
    _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x:images, y:labels})
    if i%10 == 0:
        


        print(accuracy_val)





