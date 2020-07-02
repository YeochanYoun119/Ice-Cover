'''
Title:           Ice Cover Regression
Files:           ice_cover.py
Course:          CS540, Spring 2020

Author:          Yeochan Youn
Email:           yyoun5@wisc.edu
'''

import math
import random

def get_dataset():
    ret = [[1855, 118], [1856, 151], [1857, 121], [1858, 96], [1859, 110], [1860, 117], [1861, 132], [1862, 104],
           [1863, 125], [1864, 118], [1865, 125], [1866, 123], [1867, 110], [1868, 127], [1869, 131], [1870, 99],
           [1871, 126], [1872, 144], [1873, 136], [1874, 126], [1875, 91], [1876, 130], [1877, 62], [1878, 112],
           [1879, 99], [1880, 161], [1881, 78], [1882, 124], [1883, 119], [1884, 124], [1885, 128], [1886, 131],
           [1887, 113], [1888, 88], [1889, 75], [1890, 111], [1891, 97], [1892, 112], [1893, 101], [1894, 101],
           [1895, 91], [1896, 110], [1897, 100], [1898, 130], [1899, 111], [1900, 107], [1901, 105], [1902, 89],
           [1903, 126], [1904, 108], [1905, 97], [1906, 94], [1907, 83], [1908, 106], [1909, 98], [1910, 101],
           [1911, 108], [1912, 99], [1913, 88], [1914, 115], [1915, 102], [1916, 116], [1917, 115], [1918, 82],
           [1919, 110], [1920, 81], [1921, 96], [1922, 125], [1923, 104], [1924, 105], [1925, 124], [1926, 103],
           [1927, 106], [1928, 96], [1929, 107], [1930, 98], [1931, 65], [1932, 115], [1933, 91], [1934, 94],
           [1935, 101], [1936, 121], [1937, 105], [1938, 97], [1939, 105], [1940, 96], [1941, 82], [1942, 116],
           [1943, 114], [1944, 92], [1945, 98], [1946, 101], [1947, 104], [1948, 96], [1949, 109], [1950, 122],
           [1951, 114], [1952, 81], [1953, 85], [1954, 92], [1955, 114], [1956, 111], [1957, 95], [1958, 126],
           [1959, 105], [1960, 108], [1961, 117], [1962, 112], [1963, 113], [1964, 120], [1965, 65], [1966, 98],
           [1967, 91], [1968, 108], [1969, 113], [1970, 110], [1971, 105], [1972, 97], [1973, 105], [1974, 107],
           [1975, 88], [1976, 115], [1977, 123], [1978, 118], [1979, 99], [1980, 93], [1981, 96], [1982, 54],
           [1983, 111], [1984, 85], [1985, 107], [1986, 89], [1987, 87], [1988, 97], [1989, 93], [1990, 88], [1991, 99],
           [1992, 108], [1993, 94], [1994, 74], [1995, 119], [1996, 102], [1997, 47], [1998, 82], [1999, 53],
           [2000, 115], [2001, 21], [2002, 89], [2003, 80], [2004, 101], [2005, 95], [2006, 66], [2007, 106],
           [2008, 97], [2009, 87], [2010, 109], [2011, 57], [2012, 87], [2013, 117], [2014, 91], [2015, 62], [2016, 65],
           [2017, 94], [2018, 86], [2019, 70]]
    return ret

def print_stats(dataset):
    '''
    takes the dataset as produced by the previous function and prints several statistics
    about the data; does not return anything

    :param dataset: dataset
    '''

    print(len(dataset)) # print lenth of dataset
    mean = 0
    for i in dataset:
        mean += i[1]
    mean = mean/len(dataset)
    print('{:.2f}'.format(mean)) # print mean value of data

    dev = 0
    for i in dataset:
        dev += ((i[1] - mean)**2)
    dev /= (len(dataset))-1
    dev = math.sqrt(dev)
    print('{:.2f}'.format(dev)) # print deviant


def regression(beta_0, beta_1):
    '''
    calculates and returns the mean squared error on the dataset given fixed betas
    :param beta_0: beta 0
    :param beta_1: beta 1
    :return: mean squared error
    '''
    dataset = get_dataset()
    mse = 0
    for i in dataset:
        mse += (beta_0 + beta_1*i[0] - i[1])**2
    mse /= len(dataset)
    return mse


def gradient_descent(beta_0, beta_1):
    '''
    performs a single step of gradient descent on the MSE and returns the derivative values as a tuple
    :param beta_0: beta 0
    :param beta_1: beta 1
    :return:  derivative values
    '''
    dataset = get_dataset()
    b0 = 0
    b1 = 0

    for i in dataset:
        b0 += (beta_0 + beta_1*i[0] - i[1])
        b1 += (beta_0 + beta_1*i[0] - i[1]) * i[0]
    b0 = b0 * (2 / len(dataset))
    b1 = b1 * (2 / len(dataset))
    return b0, b1


def iterate_gradient(T, eta):
    '''
    performs T iterations of gradient descent starting at LaTeX: (\beta_0, \beta_1) = (0,0)( β 0 , β 1 ) = ( 0 , 0 )
    with the given parameter and prints the results; does not return anything
    :param T: number of iterations
    :param eta: eta to calculate gradient
    '''
    b0,b1 = 0,0
    # b1 = 0
    for i in range(T):
        ite = i+1
        a,b = gradient_descent(b0, b1)
        b0, b1  = b0 - eta*a, b1 - eta*b
        mse = regression(b0, b1)
        print('{:d} {:.2f} {:.2f} {:.2f}'.format(ite, b0, b1, mse))

def compute_betas():
    '''
    using the closed-form solution, calculates and returns the values of beta_0 and beta_1
    and the corresponding MSE as a three-element tuple
    :return: calculated beta_0, calculated beta_1, mse
    '''
    dataset = get_dataset()
    ave_x = sum([i[0] for i in dataset])/ len(dataset)
    ave_y = sum([i[1] for i in dataset])/ len(dataset)

    up, down = 0,0
    for j in dataset:
        up += (j[0]-ave_x)*(j[1] - ave_y)
        down += ((j[0] - ave_x)**2)
    beta_1 = up/ down
    beta_0 = ave_y - beta_1*ave_x
    mse = regression(beta_0, beta_1)
    return beta_0, beta_1, mse


def predict(year):
    '''
    using the closed-form solution betas, return the predicted number of ice days for that year
    :param year: year which we want to know
    :return: predicted days of freeze
    '''
    beta_0, beta_1, mse = compute_betas()
    pre = beta_0 + beta_1*year
    pre = round(pre, 2)
    return pre

def iterate_normalized(T, eta):
    '''
    normalizes the data before performing gradient descent, prints results
    :param T: number of iteration
    :param eta: eta to calculate
    '''
    dataset = get_dataset()

    ave_x = sum([i[0] for i in dataset])/ len(dataset)

    std = math.sqrt(sum([(i[0]-ave_x)**2 for i in dataset]) / (len(dataset)-1))
    ls_x = [(i[0]-ave_x)/std for i in dataset]

    for k in range(len(dataset)):
        dataset[k][0] = ls_x[k]

    b0, b1 = 0,0
    for i in range(T):
        ite = i + 1
        a,b = gradient_descent_normalized(b0, b1, dataset)
        b0, b1 = b0 - eta*a, b1 - eta*b
        mse = regression_normalized(b0, b1, dataset)
        print('{:d} {:.2f} {:.2f} {:.2f}'.format(ite, b0, b1, mse))


def regression_normalized(beta_0, beta_1, dataset):
    '''
    regression for normalized
    :param beta_0: beta_0
    :param beta_1: beta_1
    :param dataset: normalized dataset
    :return: mse for normalized
    '''
    mse = 0
    for i in dataset:
        mse += (beta_0 + beta_1 * i[0] - i[1]) ** 2
    mse /= len(dataset)
    mse = round(mse, 2)
    return mse


def gradient_descent_normalized(beta_0, beta_1, dataset):
    '''
    calculate gradient for normalized
    :param beta_0: beta_0
    :param beta_1: beta_1
    :param dataset: normalized dataset
    :return: derivative values for normalized
    '''
    b0 = 0
    b1 = 0

    for i in dataset:
        b0 += (beta_0 + beta_1*i[0] - i[1])
        b1 += (beta_0 + beta_1*i[0] - i[1]) * i[0]
    b0 = b0 * (2 / len(dataset))
    b1 = b1 * (2 / len(dataset))
    return b0, b1



def sgd(T, eta):
    '''
     performs stochastic gradient descent, prints results as in function 5
    :param T: number of iteration
    :param eta: eta for calculation
    '''
    dataset = get_dataset()
    ave_x = sum([i[0] for i in dataset]) / len(dataset)
    std = math.sqrt(sum([(i[0] - ave_x) ** 2 for i in dataset]) / (len(dataset) - 1))
    ls_x = [(i[0] - ave_x) / std for i in dataset]

    for k in range(len(dataset)):
        dataset[k][0] = ls_x[k]

    b0, b1 = 0,0
    for i in range(T):
        ite = i + 1
        b0, b1 = b0 - eta*((gradient_descent_modified(b0, b1, dataset))[0]), b1 - eta*((gradient_descent_modified(b0, b1, dataset))[1])
        mse = regression_normalized(b0, b1, dataset)
        print('{:d} {:.2f} {:.2f} {:.2f}'.format(ite, b0, b1, mse))


def gradient_descent_modified(beta_0, beta_1, dataset):
    '''
    calculate gradient for sgd
    :param beta_0: beta_0
    :param beta_1: beta_1
    :param dataset: normalized dataset
    :return: derivative values for sgd
    '''
    b0, b1 = 0, 0
    pick = random.randint(0, len(dataset)-1)
    x = dataset[pick][0]
    y = dataset[pick][1]
    b0 = 2 * (beta_0 + beta_1 * x - y)
    b1 = 2 * (beta_0 + (beta_1 * x) - y) * x
    return b0, b1

data = get_dataset()
print_stats(data)