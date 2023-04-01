import random
import math
import copy
import sys


def fitness_rastrigin(position):
    fitness_value = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return fitness_value


def fitness_sphere(position):
    fitness_value = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value += (xi * xi)
    return fitness_value


class Student:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)
        self.position = [0.0 for i in range(dim)]
        for i in range(dim):
            self.position[i] = ((maxx - minx) *
                                self.rnd.random() + minx)
        self.fitness = fitness(self.position)


def tlbo(fitness, max_iter, n, dim, minx, maxx):
    rnd = random.Random(0)

    # create n random students
    classroom = [Student(fitness, dim, minx, maxx, i) for i in range(n)]

    # compute the value of best_position and best_fitness in the classroom
    Xbest = [0.0 for i in range(dim)]
    Fbest = sys.float_info.max

    for i in range(n):  # check each Student
        if classroom[i].fitness < Fbest:
            Fbest = classroom[i].fitness
            Xbest = copy.copy(classroom[i].position)

    # main loop of tlbo
    Iter = 0
    while Iter < max_iter:

        # after every 10 iterations
        # print iteration number and best fitness value so far
        if Iter % 10 == 0 and Iter > 1:
            print("Iter = " + str(Iter) + " best fitness = %.3f" % Fbest)

        # for each student of classroom
        for i in range(n):

            ### Teaching phase of ith student

            # compute the mean of all the students in the class
            Xmean = [0.0 for i in range(dim)]
            for k in range(n):
                for j in range(dim):
                    Xmean[j] += classroom[k].position[j]

            for j in range(dim):
                Xmean[j] /= n;

            # initialize new solution
            Xnew = [0.0 for i in range(dim)]

            # teaching factor (TF)
            # either 1 or 2 ( randomly chosen)
            TF = random.randint(1, 3)

            # best student of the class is teacher
            Xteacher = Xbest

            # compute new solution
            for j in range(dim):
                Xnew[j] = classroom[i].position[j] + rnd.random() * (Xteacher[j] - TF * Xmean[j])

            # if Xnew < minx OR Xnew > maxx
            # then clip it
            for j in range(dim):
                Xnew[j] = max(Xnew[j], minx)
                Xnew[j] = min(Xnew[j], maxx)

            # compute fitness of new solution
            fnew = fitness(Xnew)

            # if new solution is better than old
            # replace old with new solution
            if (fnew < classroom[i].fitness):
                classroom[i].position = Xnew
                classroom[i].fitness = fnew

            # update best student
            if (fnew < Fbest):
                Fbest = fnew
                Xbest = Xnew

            ### learning phase of ith student

            # randomly choose a solution from classroom
            # chosen solution should not be ith student
            p = random.randint(0, n - 1)
            while (p == i):
                p = random.randint(0, n - 1)

            # partner solution
            Xpartner = classroom[p]

            Xnew = [0.0 for i in range(dim)]
            if (classroom[i].fitness < Xpartner.fitness):
                for j in range(dim):
                    Xnew[j] = classroom[i].position[j] + rnd.random() * (
                                classroom[i].position[j] - Xpartner.position[j])
            else:
                for j in range(dim):
                    Xnew[j] = classroom[i].position[j] - rnd.random() * (
                                classroom[i].position[j] - Xpartner.position[j])

            # if Xnew < minx OR Xnew > maxx
            # then clip it
            for j in range(dim):
                Xnew[j] = max(Xnew[j], minx)
                Xnew[j] = min(Xnew[j], maxx)

            # compute fitness of new solution
            fnew = fitness(Xnew)

            # if new solution is better than old
            # replace old with new solution
            if (fnew < classroom[i].fitness):
                classroom[i].position = Xnew
                classroom[i].fitness = fnew

            # update best student
            if (fnew < Fbest):
                Fbest = fnew
                Xbest = Xnew

        Iter += 1
    # end-while

    # return best student from classroom
    return Xbest


print("\nBegin  teaching learning based optimization on rastrigin function\n")
dim = 3
fitness = fitness_rastrigin

print("Goal is to minimize Rastrigin's function in " + str(dim) + " variables")
print("Function has known min = 0.0 at (", end="")
for i in range(dim - 1):
    print("0, ", end="")
print("0)")

num_particles = 50
max_iter = 100

print("Setting num_particles = " + str(num_particles))
print("Setting max_iter    = " + str(max_iter))
print("\nStarting TLBO algorithm\n")

best_position = tlbo(fitness, max_iter, num_particles, dim, -10.0, 10.0)

print("\nTLBO completed\n")
print("\nBest Student found:")
print(["%.6f" % best_position[k] for k in range(dim)])
fitness_value = fitness(best_position)
print("fitness of best Student = %.6f" % fitness_value)

print("\nEnd TLBO for rastrigin function\n")

print()
print()

# Driver code for Sphere function
print("\nBegin teaching learning based optimization on sphere function\n")
dim = 3
fitness = fitness_sphere

print("Goal is to minimize sphere function in " + str(dim) + " variables")
print("Function has known min = 0.0 at (", end="")
for i in range(dim - 1):
    print("0, ", end="")
print("0)")

num_particles = 500
max_iter = 30

print("Setting num_particles = " + str(num_particles))
print("Setting max_iter    = " + str(max_iter))
print("\nStarting TLBO algorithm\n")

best_position = tlbo(fitness, max_iter, num_particles, dim, -1000.0, 1000.0)

print("\nTLBO completed\n")
print("\nBest Student found:")
print(["%.6f" % best_position[k] for k in range(dim)])
fitness_value = fitness(best_position)
print("fitness of best Student = %.6f" % fitness_value)

print("\nEnd TLBO for sphere function\n")