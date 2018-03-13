import sys
import numpy as np
import matplotlib.pyplot as plt


def estimate_mu(x, start, stop, step):
  min_diff = sys.maxsize
  best_mu = sys.maxsize

  f_mu = []
  mu_list = []

  for mu in np.arange(start, stop, step):
    mu_list.append(mu)
    z = 1.0 / np.sum([(abs(x_i - mu))**2.0 for x_i in x])
    f_mu.append(np.sum([x_i / ((abs(x_i - mu))**2.0) for x_i in x])/z)
    diff = abs(mu - f_mu[-1])
    if diff < min_diff:
      min_diff = diff
      best_mu = mu

  plt.scatter(mu_list, f_mu)
  plt.show()
  return best_mu


def _p_gaussian(x, mu):
  z = 1.0 / np.sqrt(2 * np.pi * abs(x-mu)**2)
  return z * np.exp(-(x-mu)**2/(2*abs(x-mu)**2))


def plot_prob(x):
  f_mu = []
  mu_list = []
  for mu in np.arange(9.5, 10.5, 0.001):
    mu_list.append(mu)
    f_mu.append(np.prod([_p_gaussian(x_i, mu) for x_i in x]))
  idx_max = np.argmax(f_mu)
  print("Maximum Probability: %.5f" % (mu_list[idx_max]))
  plt.scatter(mu_list, f_mu)
  plt.ylim(-.05,.3)
  plt.show()



def main():
  x = [-27.02, 3.57, 8.191, 9.898, 9.603, 9.945, 10.056]
  # mu = estimate_mu(x, -20., 20., 0.001)
  # print("mu is %.3f" % mu)
  # for i, x_i in enumerate(x):
  #   print("x[%d]: sigma_%d = %.3f" % (i, i, abs(x_i - mu)))
  plot_prob(x)


if __name__ == "__main__":
  main()
