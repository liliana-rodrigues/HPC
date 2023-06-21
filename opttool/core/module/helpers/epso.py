import sys
import random


class Particle:
    def __init__(self, x_min, x_max):
        self.dim = len(x_min)       # number of variables
        self.x_min = []             # particle min position
        self.x_max = []             # particle max position
        self.v_min = []             # particle min velocity
        self.v_max = []             # particle max velocity
        self.weights = []           # weights movement equation
        self.position_i = []        # particle position
        self.velocity_i = []        # particle velocity
        self.best_pos_i = []        # best position individual
        self.fit_pos_i = sys.float_info.max           # fit position
        self.fit_best_pos_i = sys.float_info.max      # best fit individual

        for i in range(0, self.dim):
            self.x_min.append(x_min[i])
            self.x_max.append(x_max[i])
            self.v_min.append(-(x_max[i]-x_min[i]))
            self.v_max.append((x_max[i]-x_min[i]))
            self.velocity_i.append(random.uniform(self.v_min[i], self.v_max[i]))
            self.position_i.append(random.uniform(self.x_min[i], self.x_max[i]))

        for i in range(0, 4):
            self.weights.append(random.uniform(0, 1))

    # evaluate current fitness
    def evaluate(self, costFunc, params=None, opt=None):
        self.fit_pos_i = costFunc(self.position_i, params, opt)

        # check to see if the current position is an individual best
        if self.fit_pos_i < self.fit_best_pos_i:
            self.best_pos_i = self.position_i
            self.fit_best_pos_i = self.fit_pos_i

    # update new particle velocity
    def update_velocity(self, communication_probability, g_best_pos):
        w1 = self.weights[0]    # inertia weight
        w2 = self.weights[1]    # cognitive weight
        w3 = self.weights[2]    # social weight
        w4 = self.weights[3]    # perturbation weight

        for i in range(0, self.dim):
            vel_cognitive = w2 * (self.best_pos_i[i] - self.position_i[i])
            vel_social = w3 * (((1 + w4 * random.gauss(0, 1)) * g_best_pos[i]) - self.position_i[i])
            if random.uniform(0, 1) < communication_probability:
                self.velocity_i[i] = w1 * self.velocity_i[i] + vel_cognitive + vel_social
            else:
                self.velocity_i[i] = w1 * self.velocity_i[i] + vel_cognitive

            # adjust maximum velocity if necessary
            if self.velocity_i[i] > self.v_max[i]:
                self.velocity_i[i] = self.v_max[i]

            # adjust minimum velocity if necessary
            if self.velocity_i[i] < self.v_min[i]:
                self.velocity_i[i] = self.v_min[i]

    # update the particle position based off new velocity updates
    def update_position(self):
        for i in range(0, self.dim):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i] > self.x_max[i]:
                self.position_i[i] = self.x_max[i]
                self.velocity_i[i] = -self.velocity_i[i]    # bounce

            # adjust minimum position if necessary
            if self.position_i[i] < self.x_min[i]:
                self.position_i[i] = self.x_min[i]
                self.velocity_i[i] = -self.velocity_i[i]    # bounce

            # re-check maximum velocity if necessary (case of asymmetric limits)
            if self.velocity_i[i] > self.v_max[i]:
                self.velocity_i[i] = self.v_max[i]

            # re-check minimum position if necessary (case of asymmetric limits)
            if self.velocity_i[i] < self.v_min[i]:
                self.velocity_i[i] = self.v_min[i]

    # update the particle weights
    def mutate_weights(self, mutation_rate):

        for i in range(0, 4):
            self.weights[i] = self.weights[i] + random.gauss(0, 1) * mutation_rate
            if self.weights[i] < 0:
                self.weights[i] = 0
            if self.weights[i] > 1:
                self.weights[i] = 1
