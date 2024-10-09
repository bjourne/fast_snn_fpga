from humanize import metric
from mysnn import *
from mysnn.pod2014 import *
import numpy as np

# NumPy setup
np.set_printoptions(precision=6, suppress=True)

class MyProgress:
    def __init__(self, progress, n_steps):
        self.progress = progress
        fmt = 'Building network %s / %s'
        self.top_task = self.add_task(n_steps, fmt, SCALE_N, SCALE_K)
        self.current_task = None

    def format(self, fmt, *args):
        args = tuple(['[blue]%s[/blue]' % arg for arg in args])
        return fmt % args

    def next_phase(self, fmt, *args):
        description = self.format(fmt, *args)
        self.progress.update(
            self.top_task, description = description, advance = 1)

    def next_step(self, fmt, *args):
        self.advance_task(self.current_task, 1, fmt, *args)

    def next_steps(self, advance, fmt, *args):
        self.advance_task(self.current_task, advance, fmt, *args)

    def set_current_task(self, total, fmt, *args):
        self.current_task = self.add_task(total, fmt, *args)

    # Internal
    def add_task(self, total, fmt, *args):
        name = self.format(fmt, *args)
        return self.progress.add_task(name = name,
                                      total = total,
                                      description = '')

    def advance_task(self, task, advance, fmt, *args):
        description = self.format(fmt, *args)
        self.progress.update(
            task, description = description, advance = advance)
