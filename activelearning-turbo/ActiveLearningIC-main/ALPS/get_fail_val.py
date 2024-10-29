import numpy as np
# Determine the fail_val
file_path = 'demo.sp.measure'
with open(file_path, 'r') as file:
    lines = file.readlines()
ring_delayb_0_data = []
for line in lines:
    if 'ring_delayb_0' in line:
        value = float(line.split('=')[1].strip())
        ring_delayb_0_data.append(value)
ring_delayb_0_data = np.array(ring_delayb_0_data)
fail_val = np.percentile(ring_delayb_0_data, 1)
print('The fail_val is:',fail_val)
