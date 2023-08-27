import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import math
import numpy as np
import random
import tensorflow.compat.v1 as tf

tf.set_random_seed(0)
np.random.seed(0)
random.seed(0)

def fitness(pos):
    result = tf.sqrt(pos[0] * pos[0] + pos[1] * pos[1])
    return result

class Student:
    def __init__(self, pos):
        self.pos = pos
        self.cost = fitness(pos)
        print_tf = tf.Print(self.cost, [self.cost], "Result: ")
        with tf.Session() as sess:
            print("Result: ", sess.run(print_tf))

if __name__ == "__main__":
    nVar = 2
    VarSize = (1, nVar)
    VarMin = -100
    VarMax = 100
    MaxItr = 1000
    nPop = 100

    wstart = 0.9
    wend = 0.2
    Xup = np.asarray((100, 100))
    Xlow = np.asarray((-100, -100))

    tf.disable_v2_behavior()
    tf.reset_default_graph()

    # Create TensorFlow placeholders for input variables
    pos_ph = tf.placeholder(tf.float32, shape=(None, nVar))
    print("test0")

    # Create the initial population
    pop = []
    for i in range(nPop):
        pos = np.random.randint(VarMin, VarMax, size=(1, nVar))
        pop.append(Student(pos_ph))
