import numpy as np

x = []
y = []
xs = np.array([])
ys = np.array([])

for i in range(1, 100000000):
    x.append(i)
    y.append(i**2)

xs = np.append(xs, [x])
ys = np.append(ys, [y])
mean_xsdata = np.mean(xs)
mean_ysdata = np.mean(ys)
stdxs = np.std(xs)
stdys = np.std(ys)
for k in range(len(xs)):
    xs[k] = (xs[k] - mean_xsdata) / stdxs
    ys[k] = (ys[k] - mean_ysdata) / stdys

print(xs)
print(ys)
print(mean_xsdata)
print(mean_ysdata)
print(stdxs)
print(stdys)
