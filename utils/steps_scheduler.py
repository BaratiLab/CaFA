import numpy as np
import torch
import random

# [1, 2, 4, 8]
# [0, 160000, 240000, 280000]  # 320000
class CurriculumSampler:
    # this supports progressive sampling of scalar values
    # the curriculum is step
    def __init__(self,
                 values,           # a list of values, mush have same length as milestone
                 milestone,        # a list,for example, [0, 10000, 20000] mile stone for curriculum
                 ):

        self.values = values
        self.milestone = milestone
        # create a map, mapping from milestone to corresponding value
        self.milestone2value = dict(zip(milestone, values))

        print('Milestone of curriculum: ', self.milestone,
              'Start value: ', self.values[0],
              'End value: ', self.values[-1], '')
        assert len(self.values) == len(self.milestone), 'values and milestone must have same length'

        self.current_ms_pointer = 0

    def get_value(self, global_step):
        milestone_changed = False
        # locate the range of global_step according to milestone
        if self.current_ms_pointer < len(self.milestone) - 1 and \
                global_step >= self.milestone[self.current_ms_pointer + 1]:
            self.current_ms_pointer += 1
            milestone_changed = True
        if self.current_ms_pointer >= len(self.milestone) - 1:
            self.current_ms_pointer = len(self.milestone) - 1

        ms = self.milestone[self.current_ms_pointer]

        current_value = self.milestone2value[ms]
        return current_value, milestone_changed

    def reset(self):
        self.current_ms_pointer = 0
