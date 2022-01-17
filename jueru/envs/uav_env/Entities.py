#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time




class Agent():
    def __init__(self, color = (0,200,0), size = 5):
        self.color = color
        self.size = size  # m, radius of a cycle

    def reset(self):
        self.pose = [0, 0]


class Obstacle:
    def __init__(self, world_size, number_obstacle, fixed = True):
        self.world_size = world_size
        self.number = number_obstacle
        self.shape = ['circle', 'rect']
        if not fixed:
            self.shape_list, self.pos_list, self.size_list = self.generate_obstacle()
        else:
            self.shape_list, self.pos_list, self.size_list = self.fixed_obstacle()


    def generate_obstacle(self):
        pos = []
        size = []
        shape = []
        # for i in range(self.number):
        #
        #     pos.append(np.random.randint(0, self.world_size-1, 2).tolist())
        #     a = np.random.uniform(0, 1)
        #     if a < 0.25:
        #         shape.append(self.shape[0])
        #         size.append(np.random.randint(5, 20))
        #     else:
        #         shape.append(self.shape[1])
        #         size.append(np.random.randint(8, 40, 2).tolist())
        return shape, pos, size

    def fixed_obstacle(self):
        shape_list = ['rect', 'circle', 'rect', 'rect', 'rect']
        pos_list = [[158, 169], [97, 90], [112, 67], [141, 157], [177, 75]]
        size_list = [[30, 17], 15, [18, 15], [25, 25], [15, 19]]
        shape = []
        pos = []
        size = []
        for i in range(len(shape_list)):
            shape.append(np.array(shape_list[i]))
            pos.append(np.array(pos_list[i]))
            size.append(np.array(size_list[i]))
        return shape, pos, size


#     def save(self, remark=''):
#         timestr = time.strftime("%Y%m%d_%H%M%S")
#         # filename = 'layout/' + remark + '_furniture_' + timestr + '.csv'
#         filename = remark + '_furniture_' + timestr + '.csv'
#         a = np.array(self.pos_list).reshape((len(self.pos_list), 2))
#         b = np.array(self.pos_list).reshape((len(self.pos_list), 2))
#         c = np.array(self.pos_list).reshape((len(self.pos_list), 2))
#         np.savetxt(fname=filename, X=a, fmt="%d", delimiter=",")
#
# # def load(self):


class Target:
    def __init__(self, world_size, number_target, fixed = True):
        self.world_size = world_size
        self.number = number_target
        self.shape = 'circle'
        if not fixed:
            self.pos_list, self.size_list = self.generate_target()
        else:
           self.pos_list, self.size_list = self.fixed_target()
        # self.pose = [0, 0, 0]  # (m, m, rad), absolute x, y and orientation of the mass center of the agent in env.

    def generate_target(self):
        pos = []
        size = []
        for i in range(self.number):
            pos.append(np.random.randint(50, self.world_size-50, 2).tolist())
            size.append(np.random.randint(3, 5))
        return pos, size
    def fixed_target(self):
        pos_list = [np.random.randint(0, self.world_size-1, 2).tolist()]
        size_list = [15]
        pos = []
        size = []
        for i in range(self.number):
            pos.append(np.array(pos_list[i]))
            size.append(np.array(size_list[i]))
        return pos, size




# if __name__ == '__main__':
#     # state = _State((22, 22))
#     # state._get_state()
#     dy = Agent()
#     plt.plot(0, 0, 'ro', label='point')
#     action = [[5, 5], [5, -5], [5, 5], [-5, 5], [5, 5], [-5, 5], [5, 5], [5, 5], [5, -5], [5, 5], [5, 5]]
#     for i in range(len(action)):
#         dy.update_tao(action[i])
#         a = dy.get_pose()
#         # print(f'current force: {action[i]} and current velocity {dy.pose}')
#     plt.show()
#     #
#     # ob = Furniture(world_size=200, number_furniture=10)
#     # print(ob.shape_list)
#     # print(ob.pos_list)
#     # print(ob.size_list)
#     #
#     # ru = Rubbish(world_size=200, number_rubbish=10)
#     # print(ru.pos_list)
#     # print(ru.size_list)
#     # pos: [(1, 0), (2,2)]
#     # shape [circle, rect]
#     # width, length  [(3,4), (2,5)]
#
#     # obs = np.dstack((obs, obs))
#     # obs = tf.convert_to_tensor(obs)
#     # print(obs)
