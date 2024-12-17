import numpy as np
import matplotlib.pyplot as plt

data_32 = np.loadtxt("spikes_32.csv", delimiter=",", skiprows=1)
data_16 = np.loadtxt("spikes_16.csv", delimiter=",", skiprows=1)
data_16_vec = np.loadtxt("spikes_16_vec.csv", delimiter=",", skiprows=1)

fig, axes = plt.subplots(3, sharex=True)
axes[0].scatter(data_32[:,0], data_32[:,1],s=1)
axes[1].scatter(data_16[:,0], data_16[:,1],s=1)
axes[2].scatter(data_16_vec[:,0], data_16_vec[:,1],s=1)

print(f"32-bit average spikes per neuron={np.average(np.bincount(data_32[:,1].astype(int)))}")
print(f"16-bit average spikes per neuron={np.average(np.bincount(data_16[:,1].astype(int)))}")
print(f"16-bit vec average spikes per neuron={np.average(np.bincount(data_16_vec[:,1].astype(int)))}")
plt.show()
