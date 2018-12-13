# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 14:05:25 2018

@author: bdenn
"""

p_theta = np.poly1d(np.polyfit(theta[20:50], theta_chunk, 2))
p_r = np.poly1d(np.polyfit(r[2:8], r_chunk, 2))

plt.figure(1)
plt.plot(theta[20:50], theta_chunk, '.', theta[20:50], p_theta(theta[20:50]))

plt.figure(2)
plt.plot(r[2:8], r_chunk, '.', r[2:8], p_r(r[2:8]))
plt.show()

print(p_r)
print(p_theta)