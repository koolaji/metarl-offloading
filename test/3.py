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
    pos_float = tf.cast(pos, dtype=tf.float32)
    result = tf.sqrt(pos_float[:, 0] * pos_float[:, 0] + pos_float[:, 1] * pos_float[:, 1])
    return result

class Student:
    def __init__(self, pos):
        self.pos = pos
        self.cost = fitness(pos)

if __name__ == "__main__":
    nVar = 2
    VarSize = (1, nVar)
    VarMin = -100
    VarMax = 100
    MaxItr = 100
    nPop = 100
    wstart = 0.9
    wend = 0.2
    Xup = np.asarray((100, 100))
    Xlow = np.asarray((-100, -100))
    tf.reset_default_graph()
    pos_ph = tf.placeholder(tf.float32, shape=(None, nVar))
    pop = []
    for i in range(nPop):
        pos = np.random.randint(VarMin, VarMax, size=(1, nVar))
        pos = pos.astype(np.float32)  
        pop.append(Student(pos))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for Itr in range(MaxItr):
            Mean = np.zeros((1, nVar))
            Teacher = pop[0]
            for k in range(nPop):
                mean_pos =  pop[k].pos  
                Mean += mean_pos
                if sess.run(pop[k].cost, feed_dict={pos_ph: pop[k].pos}) < sess.run(Teacher.cost, feed_dict={pos_ph: Teacher.pos}):
                    Teacher = pop[k]
                    print("Teacher", k, Teacher.pos, sess.run(pop[k].cost, feed_dict={pos_ph: pop[k].pos}) )
            Mean /= nPop
            for i in range(nPop):
                    rand_bar = random.random()
                    TF = random.randrange(1, 3)
                    feed_dict_teacher = {pos_ph: Teacher.pos}
                    diff = rand_bar * (sess.run(Teacher.cost, feed_dict=feed_dict_teacher) - TF * Mean)
                    w = wstart - (wstart - wend) * (Itr / MaxItr)
                    feed_dict = {pos_ph: pop[k].pos}
                    if sess.run(pop[i].cost, feed_dict=feed_dict) < sess.run(fitness(Mean), feed_dict=feed_dict):
                        newPos = pop[i].pos * w + Teacher.pos - pop[i].pos * rand_bar
                    else:
                        newPos = (pop[i].pos + (rand_bar - 0.5) * 2 * (Mean - pop[i].pos)) * math.sin(
                            (math.pi / 2) * (Itr / MaxItr)) + diff * math.cos((math.pi / 2) * (Itr / MaxItr))
                    newPos = sess.run(tf.maximum(tf.minimum(newPos, VarMax), VarMin))
                    newCost = fitness(newPos.astype(np.float32))
                    if sess.run(newCost , feed_dict={pos_ph: newPos})  < sess.run(pop[i].cost, feed_dict={pos_ph: pop[i].pos}):
                        pop[i].pos = newPos
                        pop[i].cost = newCost
            print("END OF LOOP")