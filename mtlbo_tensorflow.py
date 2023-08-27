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
        print("Result: ",print_tf)

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
        print(f"pos : {pos}")
        pop.append(Student(pos_ph))
    print("test1")
    # Initialize TensorFlow session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Main optimization loop
        for Itr in range(MaxItr):
            Mean = np.zeros((1, nVar))
            Teacher = pop[0]

            # Update Teacher and calculate the mean position
            for k in range(nPop):
                # Provide input data for the placeholder
                feed_dict = {pos_ph: np.zeros((1, nVar))}
                Mean += sess.run(pop[k].pos, feed_dict=feed_dict)
                if sess.run(pop[k].cost, feed_dict=feed_dict) < sess.run(Teacher.cost, feed_dict=feed_dict):
                    Teacher = pop[k]
                    print("Teacher", Itr, sess.run(Teacher.cost, feed_dict=feed_dict))
                print(k)

                Mean /= nPop

                # Update positions for each individual in the population
                for i in range(nPop):
                    print(i)
                    rand_bar = random.random()
                    TF = random.randrange(1, 3)
                    diff = rand_bar * (sess.run(Teacher.pos, feed_dict={pos_ph: np.zeros((1, nVar))}) - TF * Mean)
                    w = wstart - (wstart - wend) * (Itr / MaxItr)

                    # Provide input data for the placeholder
                    feed_dict = {pos_ph: np.zeros((1, nVar))}

                    if sess.run(pop[i].cost, feed_dict=feed_dict) < sess.run(fitness(Mean), feed_dict=feed_dict):
                        newPos = sess.run(pop[i].pos, feed_dict=feed_dict) * w + (
                                sess.run(Teacher.pos, feed_dict=feed_dict) - sess.run(pop[i].pos, feed_dict=feed_dict)) * rand_bar
                    else:
                        newPos = (sess.run(pop[i].pos, feed_dict=feed_dict) + (rand_bar - 0.5) * 2 * (
                                Mean - sess.run(pop[i].pos, feed_dict=feed_dict))) * tf.sin(
                            (math.pi / 2) * (Itr / MaxItr)) + diff * tf.cos((math.pi / 2) * (Itr / MaxItr))

                    # Apply variable bounds
                    newPos = tf.maximum(tf.minimum(newPos, VarMax), VarMin)

                    # Update cost and position
                    newCost = sess.run(fitness(newPos), feed_dict=feed_dict)
                    if newCost < sess.run(pop[i].cost, feed_dict=feed_dict):
                        sess.run(tf.assign(pop[i].pos, newPos), feed_dict=feed_dict)
                        sess.run(tf.assign(pop[i].cost, newCost), feed_dict=feed_dict)