import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab as pl
h = sorted([186, 176, 158, 180, 186, 168, 168, 164, 178, 170, 189, 195, 172, 187, 180, 186, 185, 168, 179, 178, 170, 175, 186, 159, 161, 178, 183, 179, 178, 179, 170, 175, 186, 159, 161, 178, 185, 187, 175, 162, 173, 172, 177, 175, 172, 177, 180])
fit = stats.norm.pdf(h, np.mean(h), np.std(h))
print(stats.norm.pdf(191, np.mean(h), np.std(h))
pl.plot(h,fit,'-o')
pl.hist(h,normed=True)
print(np.mean(h))
print(np.std(h))
pl.show()

