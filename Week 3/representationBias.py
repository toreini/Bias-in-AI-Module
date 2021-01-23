import numpy
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import random

# Returns theta in [-pi/2, 3pi/2]
def generate_theta(a, b):
    u = random.random() / 4.0
    theta = numpy.arctan(b/a * numpy.tan(2*numpy.pi*u))

    v = random.random()
    if v < 0.25:
        return theta
    elif v < 0.5:
        return numpy.pi - theta
    elif v < 0.75:
        return numpy.pi + theta
    else:
        return -theta

def radius(a, b, theta):
    return a * b / numpy.sqrt((b*numpy.cos(theta))**2 + (a*numpy.sin(theta))**2)

def random_point(a, b):
    random_theta = generate_theta(a, b)
    max_radius = radius(a, b, random_theta)
    random_radius = max_radius * numpy.sqrt(random.random())

    return numpy.array([
        random_radius * numpy.cos(random_theta),
        random_radius * numpy.sin(random_theta)
    ])

a = 2
b = 1

points1 = numpy.array([random_point(2, 1) for _ in range(1000)])

points2 = numpy.array([random_point(2, 1)+4 for _ in range(2000)])

points3 = numpy.array([random_point(2, 1)+8 for _ in range(2000)])


cov = [[3, -2], [-2, 3]]

mean_1 = [1, 1]
x_1, y_1 = numpy.random.multivariate_normal(mean_1, cov, 5000).T

mean_2 = [8, 8]
x_2, y_2 = numpy.random.multivariate_normal(mean_2, cov, 5000).T

mean_3 = [16, 16]
x_3, y_3 = numpy.random.multivariate_normal(mean_3, cov, 5000).T

plt.scatter(x_1, y_1,color='red')
plt.scatter(x_2, y_2,color='blue')
plt.scatter(x_3, y_3,color='green')


x_1_train = x_1.reshape((-1,1))
regr = linear_model.LinearRegression()
regr.fit(x_1_train, y_1)


x1_pred = numpy.linspace(-10,10,300).reshape((-1, 1))
y1_pred = regr.predict(x1_pred)

x_2_train = x_2.reshape((-1,1))
regr = linear_model.LinearRegression()
regr.fit(x_2_train, y_2)


x2_pred = numpy.linspace(-5,15,300).reshape((-1, 1))
y2_pred = regr.predict(x2_pred)

x_3_train = x_3.reshape((-1,1))
regr = linear_model.LinearRegression()
regr.fit(x_3_train, y_3)


x3_pred = numpy.linspace(0,25,300).reshape((-1, 1))
y3_pred = regr.predict(x3_pred)

plt.plot(x1_pred, y1_pred, 'm--', linewidth=2)
plt.plot(x2_pred, y2_pred, 'm--', linewidth=2)
plt.plot(x3_pred, y3_pred, 'm--', linewidth=2)

x_train=numpy.concatenate((x_1_train,x_2_train,x_3_train))
y_train=numpy.concatenate((y_1,y_2,y_3))

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

x_pred = numpy.linspace(-10,25,300).reshape((-1, 1))
y_pred = regr.predict(x_pred)

plt.plot(x_pred, y_pred, 'c--',linewidth=2)

plt.show()