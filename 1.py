import tensorflow as tf
print(tf.test.is_gpu_available())
print(tf.test.gpu_device_name())
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
