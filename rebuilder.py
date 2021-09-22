from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from random import randint
import matplotlib.pyplot as plt

print("loading dataset...")
print("It may take some seconds")
X, Y = fetch_openml('mnist_784', version=1, return_X_y=True)
print("dataset loaded")

x = []
for i in range(10):
    r = randint(1, len(X) / 2)
    j = r
    for j in range(r, len(X)):
        if int(Y[j]) == i:
            x.append(j)
            break
    for k in range(j + 1, len(X)):
        if int(Y[k]) == i:
            x.append(k)
            break

pca_components = [30, 90, 150, 210, 270, 330, 390, 450, 510, 570]
error = []
print("possessing part a ..........", end="")
a = 1
for component in pca_components:
    pca = PCA(n_components=component)
    lower_dimensional_data = pca.fit_transform(X)
    approximation = pca.inverse_transform(lower_dimensional_data)
    plt.figure(figsize=(8, 4))
    count = 1
    counter = 0.0
    for i in x:
        plt.subplot(2, 10, count)
        plt.imshow(approximation[i].reshape(28, 28), interpolation='nearest', clim=(0, 255))
        count += 1
        c = 0.0
        for k in range(len(X[i])):
            c += (X[i][k] - approximation[i][k]) ** 2
        counter += c / len(X[i])
    error.append(counter / len(x))
    print("\b\b\b\b\b\b\b\b\b\b", end="")
    for j in range(a):
        print("*", end="")
    for j in range(10 - a):
        print(".", end="")
    a += 1
    plt.suptitle(str(component) + ' components part a', fontsize=14)

plt.figure()
plt.plot(pca_components, error)
plt.xlabel('number of components')
plt.ylabel('error')
plt.title("error for part a")

x2 = [[], [], [], [], [], [], [], [], [], []]
x3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(len(X)):
    x2[int(Y[i])].append(X[i])
    for j in range(len(x)):
        if i == x[j]:
            x3[j] = len(x2[int(Y[i])]) - 1
            break

error2 = []
print()
print("possessing part b ..........", end="")
a = 1
for component in pca_components:
    pca = PCA(n_components=component)
    plt.figure(figsize=(8, 4))
    count = 1
    counter = 0.0
    for j in range(10):
        lower_dimensional_data = pca.fit_transform(x2[j])
        approximation = pca.inverse_transform(lower_dimensional_data)
        i = x3[2 * j]
        ii = x[2 * j]
        plt.subplot(2, 10, count)
        plt.imshow(approximation[i].reshape(28, 28), interpolation='nearest', clim=(0, 255))
        count += 1
        c = 0.0
        for k in range(len(X[ii])):
            c += (X[ii][k] - approximation[i][k]) ** 2
        counter += c / len(X[ii])
        i = x3[2 * j + 1]
        ii = x[2 * j + 1]
        plt.subplot(2, 10, count)
        plt.imshow(approximation[i].reshape(28, 28), interpolation='nearest', clim=(0, 255))
        count += 1
        c = 0.0
        for k in range(len(X[ii])):
            c += (X[ii][k] - approximation[i][k]) ** 2
        counter += c / len(X[ii])
    error2.append(counter / 20)

    print("\b\b\b\b\b\b\b\b\b\b", end="")
    for j in range(a):
        print("*", end="")
    for j in range(10 - a):
        print(".", end="")
    a += 1
    plt.suptitle(str(component) + ' components part b', fontsize=14)

plt.figure()
plt.plot(pca_components, error2)
plt.xlabel('number of components')
plt.ylabel('error')
plt.title("error for part b")

plt.show()
