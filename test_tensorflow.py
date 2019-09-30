import unittest
import tensorflow as tf


class TestStringMethods(unittest.TestCase):

    def test_tensorflow(self):
        tf.enable_eager_execution()
        result = tf.add(1, 2)
        hello = tf.constant('Hello, TensorFlow!')
        hello.numpy()


if __name__ == '__main__':
    unittest.main()





