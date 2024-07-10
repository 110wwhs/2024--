import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

p = 0
mu = p 
se = 1 
sigma = se

x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
y = norm.pdf(x, mu, sigma)

ci_lower = mu - 2 * sigma
ci_upper = mu + 2 * sigma

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Normal Distribution', color='blue')

ci_x = np.linspace(ci_lower, ci_upper, 100)
ci_y = norm.pdf(ci_x, mu, sigma)
plt.fill_between(ci_x, ci_y, color='red', alpha=0.5, label='99% Confidence Interval')

plt.title('Normal Distribution of 2030 Crime Clearance Rate with 99% Confidence Interval')
plt.xlabel('Crime Clearance Rate')
plt.ylabel('Probability Density')
plt.axvline(ci_lower, color='red', linestyle='dashed')
plt.axvline(ci_upper, color='red', linestyle='dashed')
plt.legend()

plt.grid(True)
plt.show()




