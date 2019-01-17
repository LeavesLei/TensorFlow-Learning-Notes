## Dataset API (tensorflow)
Sammary of TensorFlow **Dataset API** by [Leaves](leishiye@gmail.com).
### previous format
---
there were two methods to read the data before tensorflow 1.3:
1. placehold (read data form RAM)
2. use queue to read the data in the hard disk

For Dataset API, it can read the data from both RAM and hard disk

### import
---
- before tf 1.3 : tf.contrib.data.Dataset
- After tf 1.4 : tf.data.Dataset

### Introduction to Dataset API
---
![Dataset API](https://pic2.zhimg.com/80/v2-f9f42cc5c00573f7baaa815795f1ce45_hd.jpg)

*two most important classes* : `Dataset`,  `Iterator`

`Dataset` can be thought of as **an ordered list* of the same type of elements.

Now, we can creat a simple dataset:
```python
import tensorflow as tf
import numpy as np

dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
```

So, how can we take out data from Dataset?  
use `Iterator`:
```python
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(5):
        print(sess.run(one_element))
# output: 1.0  2.0  3.0  4.0  5.0
```

if we continue to run `sess.run(one_element)` after having read out of dataset, the pragram will throw out **tf.errors.OutOfRangeError**

Above introduction is just a warm up. Now, let's continue learning about Dataset API !

### tf.data.Dataset.from_tensor_slices
---
What is the function of `tf.data.Dataset.from_tensor_slices` ?  
  **Splits each rank-N tf.SparseTensor in this dataset row-wise.**

I prefer to use a example to explain `tf.data.Dataset.from_tensor_slices`:
```python
dataset = tf.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        
        "b": np.random.uniform(size=(5, 2))
    }
)
```
A element in the dataset is `{"a": 1.0, "b": [0.9, 0.1]}` and the dataset has a total of 5 elements.

another example:
```python
dataset = tf.data.Dataset.from_tensor_slices(
  (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.random.uniform(size=(5, 2)))
)
```
A element in the dataset is `{1.0, [0.9, 0.1]}` and the dataset has a total of 5 elements

### Transformation : covert a dataset to a new dataset
---
There are several transformation operation
#### 1. map()
```python
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.map(lambda x: x + 1) # 2.0, 3.0, 4.0, 5.0, 6.0
```
#### 2. batch()
batch() is to combine multiple elements into a batch
```python
dataset = dataset.batch(32)
```
#### 3. shuffle()
shuffle() is to disturb the elements in the dataset. It has a parameter buffer_size, which indicates the size of the buffer used during the disruption.
```python
dataset = dataset.shuffle(buffer_size=10000)
```
there is a detail discription about the parameter `buffer_size` with Chinese: [explain of buffer_size](https://zhuanlan.zhihu.com/p/42417456)
#### 4. repeat()
if there are 5 epoch in your training procedure, you need to repeat the dataset 5 times
```
dataset = dataset.repeat(5)
```
**WARNING**: if you use: `dataset = dataset.repeat()`, the dataset will repeat forever.

#### 5. combied example for make a photo dataset
```python
# the function is to read images through their filename and resize them to the same size 28×28.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

# list of imamge filename
filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])
# label[i] the label of the image filenames[i]
labels = tf.constant([0, 37, ...])

# one element of the dataset is (filename, label)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

# one element of the dataset is (image_resized, label)
dataset = dataset.map(_parse_function)

# on element of the dataset is (image_resized_batch, label_batch)
dataset = dataset.shuffle(buffersize=1000).batch(32).repeat(10)
```
### Other methods of creating a dataset
---
- `tf.data.TextLineDataset()` is often used to read csv files
- `tf.data.FixedLengthRecordDataset()`
- `tf.data.TFRecordDataset()` is often used to read TFRecord files

you can find more detail about above three methods in [Module: tf.data](https://www.tensorflow.org/api_docs/python/tf/data)

### How to make a iterator
---
- use make_one_shot_iterator() to make one shot iterator
- initializable iterator
- reinitializable iterator
- feedable iterator

a example of `initializable iterator`:
```python
limit = tf.placeholder(dtype=tf.int32, shape=[])

dataset = tf.data.Dataset.from_tensor_slices(tf.range(start=0, limit=limit))

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={limit: 10})
    for i in range(10):
      value = sess.run(next_element)
      assert i == value
```

another example of `initializable iterator`:
```python
# read two numpy array from hard disk
with np.load("/var/data/training_data.npy") as data:
  features = data["features"]
  labels = data["labels"]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
iterator = dataset.make_initializable_iterator()
sess.run(iterator.initializer, feed_dict={features_placeholder: features,labels_placeholder: labels})
```

### wait to do/write
---
- more detail about iterator
- `eager model` and `not eager model`

### reference
---
- [==Dataset API introduction with Chinese==](https://zhuanlan.zhihu.com/p/30751039)
- [==tensorflow official guide==](https://www.tensorflow.org/guide/datasets)
- [==module: tf.data==](https://www.tensorflow.org/api_docs/python/tf/data)
- [==Introduction to TensorFlow Datasets and Estimators==](https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html)
