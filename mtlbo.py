import math
import numpy as np
import random

rnd = random.Random(0)


def fitness(pos):
    # print(type(pos))
    # print(pos)
    return math.sqrt(pos[0] * pos[0] + pos[1] * pos[1])


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
    wstart = 0.9;
    wend = 0.2;
    Xup = np.asarray((100, 100))
    Xlow = np.asarray((-100, -100))

    pop = []
    for i in range(0, nPop):
        pop.append(Student(np.random.randint(VarMin, VarMax, size=nVar)))

    # for i in range(0, nPop):
    #     print(pop[i].pos, pop[i].cost)

    Mean = 0
    Teacher = pop[0]
    for Itr in range(0, MaxItr):
        for k in range(0, nPop):
            Mean = pop[k].pos + Mean
            if pop[k].cost < Teacher.cost:
                Teacher = pop[k]
                print("Teacher", Itr, Teacher.cost)
        Mean = Mean / nPop
        # print(Mean)
        # print(Teacher.pos)
        for i in range(0, nPop):
            # a = (rnd.random(), rnd.random())
            rand_bar = rnd.random()
            TF = random.randrange(1, 3)
            diff = rand_bar * (Teacher.pos - TF * Mean)
            w = wstart - (wstart - wend) * (Itr / MaxItr)
            # newPos = pop[i].pos + rand_bar * (Teacher.pos - TF * Mean)
            # # print(newPos)
            if pop[i].cost < fitness(Mean):
                newPos = pop[i].pos * w + (Teacher.pos - pop[i].pos) * rand_bar
            else:
                newPos = (pop[i].pos + (rand_bar - 0.5) * 2 * (Mean - pop[i].pos)) * math \
                    .sin((math.pi / 2) * (Itr / MaxItr)) + diff * math.cos((math.pi / 2) * (Itr / MaxItr))

            if newPos[0] > VarMax:
                newPos[0] = VarMax
            if newPos[1] > VarMax:
                newPos[1] = VarMax

            if newPos[0] < VarMin:
                newPos[0] = VarMin
            if newPos[1] < VarMin:
                newPos[1] = VarMin
            # print(newPos)
            newCost = fitness(newPos)
            if newCost < pop[i].cost:
                pop[i].pos = newPos
                pop[i].cost = newCost
                # print(i)
                # print(pop[i].cost)
        # for i in range(0, nPop):
        #     print("hasan1",pop[i].pos, pop[i].cost)
        pop.sort(key=lambda x: x.cost)
        # print()
        # for i in range(0, nPop):
        #     print("hasan",pop[i].pos, pop[i].cost)
        for j in range(0, int(nPop/2)):
            p = random.randint(0, nPop/2 - 1)
            while p == j:
                p = random.randint(0, nPop/2 - 1)
            rand_bar = rnd.random()
            if pop[j].cost > pop[p].cost :
                newPos = pop[j].pos + (pop[p].pos - pop[j].pos)*math.\
                    cos((math.pi/2)*(Itr/MaxItr))
            else:
                newPos = pop[j].pos + (rand_bar - 0.5)*2*(Xup - Xlow)

            if newPos[0] > VarMax:
                newPos[0] = VarMax
            if newPos[1] > VarMax:
                newPos[1] = VarMax

            if newPos[0] < VarMin:
                newPos[0] = VarMin
            if newPos[1] < VarMin:
                newPos[1] = VarMin
            # print(newPos)
            newCost = fitness(newPos)
            if newCost < pop[j].cost:
                pop[j].pos = newPos
                pop[j].cost = newCost
        for j in range(int(nPop/2+1), nPop):
            newPos = pop[j].pos+(Teacher.pos-pop[j].pos)*math.cos((math.pi/2)*(Itr/MaxItr))
            if newPos[0] > VarMax:
                newPos[0] = VarMax
            if newPos[1] > VarMax:
                newPos[1] = VarMax

            if newPos[0] < VarMin:
                newPos[0] = VarMin
            if newPos[1] < VarMin:
                newPos[1] = VarMin
            # print(newPos)
            newCost = fitness(newPos)
            if newCost < pop[j].cost:
                pop[j].pos = newPos
                pop[j].cost = newCost


    # print("\n")
    # for i in range(0, nPop):
    #     print(pop[i].pos, pop[i].cost)
