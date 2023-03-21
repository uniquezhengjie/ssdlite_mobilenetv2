import numpy as np

learning_rate_base = 0.1
# global_step = 143000
# total_steps = 250000
global_step = 143000
total_steps = 250000
warmup_steps = 2000
learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi * (global_step - warmup_steps \
        ) / float(total_steps - warmup_steps)))

print(learning_rate)