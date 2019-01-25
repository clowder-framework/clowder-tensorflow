import tensorflow as tf

tf.enable_eager_execution()
result = tf.add(1, 2)
print(result)
hello = tf.constant('Hello, TensorFlow!')
hello.numpy()
print('done')
