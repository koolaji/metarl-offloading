import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import random

rnd = random.Random(0)


def fitness(pos):
    pos_tensor = tf.constant(pos, dtype=tf.float32)
    squared_sum = tf.reduce_sum(tf.square(pos_tensor))
    return tf.sqrt(squared_sum)


class Student:
    def __init__(self, pos):
        self.pos = pos
        self.cost = fitness(pos)


if __name__ == "__main__":

    nVar = 2
    VarSize = (1, nVar)
    VarMin = -100
    VarMax = 100
    MaxItr = 1000
    nPop = 100
    ### mtlbo
    wstart = 0.9
    wend = 0.2
    Xup = np.asarray((100, 100))
    Xlow = np.asarray((-100, -100))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        pop = []
        for i in range(0, nPop):
            pos = np.random.randint(VarMin, VarMax, size=nVar)
            student = Student(pos)
            student.cost_value = sess.run(student.cost)
            pop.append(student)
            print(f"student.pos: {student.pos}  student.cost_value {student.cost_value} student.cost {student.cost}")

        # Main optimization loop
        Teacher = pop[0]
        print(f"Teacher: {Teacher.cost_value}")
        Mean = 0
        for k in range(nPop):
                # Provide input data for the placeholder
                feed_dict = {pos_ph: np.zeros((1, nVar))}
                Mean += sess.run(pop[k].pos, feed_dict=feed_dict)
                if sess.run(pop[k].cost, feed_dict=feed_dict) < sess.run(Teacher.cost, feed_dict=feed_dict):
                    Teacher = pop[k]
                    print("Teacher", Itr, sess.run(Teacher.cost, feed_dict=feed_dict))
                print(k)
            
                Mean /= nPop