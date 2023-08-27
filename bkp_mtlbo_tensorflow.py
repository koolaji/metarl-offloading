import tensorflow as tf
import math

class Student:
    def __init__(self, pos):
        self.pos = tf.Variable(pos)  # Make pos a TensorFlow Variable
        self.cost = fitness(pos)

def fitness(pos):
    return tf.sqrt(pos[0][0] * pos[0][0] + pos[0][1] * pos[0][1])

def mtlbo():
    nVar = 2
    VarSize = (1, nVar)
    VarMin = -100
    VarMax = 100
    MaxItr = 1000
    nPop = 100
    
    pop = []
    for i in range(nPop):
        pos = tf.random_uniform(VarSize, VarMin, VarMax)
        pop.append(Student(pos))

    Mean = tf.zeros(VarSize, dtype=tf.float32)
    Teacher = pop[0]
    
    # Initialize variables
    init_op = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init_op)  # Initialize variables
        
        for Itr in range(MaxItr):
            for k in range(nPop):
                Mean = tf.add(pop[k].pos, Mean)
                cond = tf.less(pop[k].cost, Teacher.cost)
                assign_op = Teacher.pos.assign(tf.where(cond, pop[k].pos, Teacher.pos))  # Assign new Teacher if condition is satisfied
                sess.run(assign_op)
                print("Teacher", Itr, str(Teacher.cost))
            Mean = tf.divide(Mean, nPop)

            for i in range(nPop):
                rand_bar = tf.random_uniform(())
                TF = tf.random_uniform((), 1, 3, dtype=tf.int32)
                diff = tf.multiply(rand_bar, tf.subtract(Teacher.pos, tf.multiply(tf.cast(TF, tf.float32), Mean)))
                wstart = tf.constant(0.9)
                wend = tf.constant(0.2)
                w = tf.subtract(wstart, tf.multiply(tf.subtract(wstart, wend), tf.divide(tf.cast(Itr, tf.float32), MaxItr)))

                cond = tf.less(pop[i].cost, fitness(Mean))
                newPos = tf.where(cond, tf.add(tf.multiply(pop[i].pos, w), tf.multiply(tf.subtract(pop[i].pos, Mean), rand_bar)),
                                  tf.add(tf.add(tf.multiply(pop[i].pos, w), tf.multiply(tf.subtract(pop[i].pos, Mean), rand_bar)), tf.multiply(diff, tf.cos(tf.multiply(tf.constant(math.pi / 2), tf.divide(tf.cast(Itr, tf.float32), MaxItr))))))

                # newPos = tf.where(tf.greater(newPos, tf.cast(VarMax, tf.float32)), tf.cast(VarMax, tf.float32), newPos)  # Cast VarMax to float32
                newPos = tf.where(tf.less(newPos, tf.cast(VarMin, tf.float32)), tf.fill(tf.shape(newPos), tf.cast(VarMin, tf.float32)), newPos)
                # newPos = tf.where(tf.less(newPos, tf.cast(VarMin, tf.float32)), tf.cast(VarMin, tf.float32), newPos)  # Cast VarMin to float32
                newPos = tf.where(tf.less(newPos, tf.cast(VarMin, tf.float32)), tf.tile(tf.reshape(tf.cast(VarMin, tf.float32), [1, 1]), [1, nVar]), newPos)

                newCost = fitness(newPos)
                assign_op = pop[i].pos.assign(newPos)
                sess.run(assign_op)
                pop[i].cost = sess.run(newCost)

            pop.sort(key=lambda x: sess.run(tf.convert_to_tensor(x.cost)).tolist())


            for j in range(nPop // 2):
                p = tf.random_uniform((), 0, nPop // 2, dtype=tf.int32)
                p = tf.cond(tf.equal(p, j), lambda: tf.random_uniform((), 0, nPop // 2, dtype=tf.int32), lambda: p)
                rand_bar = tf.random_uniform(())
                # cond = tf.greater(pop[j].cost, pop[p].cost)
                # cond = tf.greater(pop[tf.cast(j, tf.int32)].cost, pop[tf.cast(p, tf.int32)].cost)
                pop_tensors = [student.cost for student in pop]
                cond = tf.greater(tf.gather(pop_tensors, tf.squeeze(tf.cast(j, tf.int32))), tf.gather(pop_tensors, tf.squeeze(tf.cast(p, tf.int32))))

                pop_tensors = [tf.convert_to_tensor(student.pos) for student in pop]
                newPos = tf.where(cond, tf.add(tf.gather(pop_tensors, tf.squeeze(tf.cast(j, tf.int32)), axis=0), tf.multiply(tf.subtract(tf.gather(pop_tensors, tf.squeeze(tf.cast(p, tf.int32)), axis=0), tf.gather(pop_tensors, tf.squeeze(tf.cast(j, tf.int32)), axis=0)), tf.cos(tf.multiply(tf.constant(math.pi / 2), tf.divide(tf.cast(Itr, tf.float32), MaxItr))))),
                                  tf.add(tf.gather(pop_tensors, tf.squeeze(tf.cast(j, tf.int32)), axis=0), tf.multiply(tf.subtract(rand_bar, 0.5), tf.multiply(2.0, tf.subtract(tf.cast(VarMax, tf.float32), tf.cast(VarMin, tf.float32))))))
                newPos = tf.where(tf.greater(newPos, tf.cast(VarMax, tf.float32)), tf.fill(tf.shape(newPos), tf.cast(VarMax, tf.float32)), newPos)
                newPos = tf.where(tf.less(newPos, tf.cast(VarMin, tf.float32)), tf.fill(tf.shape(newPos), tf.cast(VarMin, tf.float32)), newPos)

                newCost = fitness(newPos)
                assign_op = pop[j].pos.assign(newPos)
                sess.run(assign_op)
                pop[j].cost = sess.run(newCost)

            for j in range(nPop // 2 + 1, nPop):
                newPos = tf.add(pop[j].pos, tf.multiply(tf.subtract(Teacher.pos, pop[j].pos), tf.cos(tf.multiply(tf.constant(math.pi / 2), tf.divide(tf.cast(Itr, tf.float32), MaxItr)))))

                newPos = tf.where(tf.greater(newPos, tf.cast(VarMax, tf.float32)), tf.broadcast_to(tf.cast(VarMax, tf.float32), tf.shape(newPos)), newPos)
                newPos = tf.where(tf.less(newPos, tf.cast(VarMin, tf.float32)), tf.broadcast_to(tf.cast(VarMin, tf.float32), tf.shape(newPos)), newPos)
                newCost = fitness(newPos)
                assign_op = pop[j].pos.assign(newPos)
                sess.run(assign_op)
                pop[j].cost = sess.run(newCost)
        
        return sess.run(Teacher.cost)

mtlbo()