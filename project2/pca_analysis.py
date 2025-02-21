from init import *
from read_data import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,FactorAnalysis, KernelPCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import seaborn as sns
import os



try:
    X
except NameError:
    init=Init()

    X=init.X
    y=init.y
    labels=init.labels


if not os.path.isdir('best_features'):

    os.mkdir('best_features')


figures_path='figures'

if not os.path.exists(figures_path):
    os.mkdir(figures_path)

## read all the saved models filenames

scaler = StandardScaler()


X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)

pca.fit(X)
X = pca.transform(X)

labels=['c-CS-m','c-SC-m','c-CS-s','c-SC-s','t-CS-m','t-SC-m','t-CS-s','t-SC-s']

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

colors='bgrcmyk2'

ax = plt.figure(figsize=(16,10)).gca(projection='3d')

for i, c,m, label in zip([0,1,2,3,4,5,6,7], 'bgrcmyk2','o+,^vx8s', labels):

    ax.scatter(xs=X[y==i,0], ys=X[y==i,1], zs=X[y==i,2], label=label,marker = m,color=c)

ax.legend()

ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')


plt.title("PCA Projection for three principal components")
plt.savefig(figures_path+os.path.sep+"PCA_3d.png")



X_reduced = pca.fit_transform(X_scaled)

plt.figure(figsize=(6, 5))
for i, c,m, label in zip([0,1,2,3,4,5,6,7], 'bgrcmyk2','o+,^vx8s', labels):
    plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1],
               c=c,marker=m, label=label)
plt.legend()

plt.gca().set_aspect("equal")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("2D PCA Projection")
plt.savefig(figures_path+os.path.sep+"PCA_12.png")

plt.close()


plt.figure(figsize=(6, 5))
for i, c,m, label in zip([0,1,2,3,4,5,6,7], 'bgrcmyk2','o+,^vx8s', labels):
    plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 2],
               c=c,marker=m, label=label)
plt.legend()

plt.xlabel("PC1")
plt.ylabel("PC3")
plt.title("2D PCA Projection ")
plt.savefig(figures_path+os.path.sep+"PCA_13.png")

plt.close()


plt.figure(figsize=(6, 5))
for i, c,m, label in zip([0,1,2,3,4,5,6,7], 'bgrcmyk2','o+,^vx8s', labels):
    plt.scatter(X_reduced[y == i, 1], X_reduced[y == i, 2],
               c=c,marker=m, label=label)
plt.legend()
plt.gca().set_aspect("equal")
plt.xlabel("PC2")
plt.ylabel("PC3")
plt.title("2D PCA Projection ")
plt.savefig(figures_path+os.path.sep+"PCA_23.png")

plt.close()


fa = FactorAnalysis(n_components=3)

X_reduced = fa.fit_transform(X_scaled)


plt.figure(figsize=(6, 5))
for i, c,m, label in zip([0,1,2,3,4,5,6,7], 'bgrcmyk2','o+,^vx8s', labels):
    plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1],
               c=c,marker=m, label=label)

plt.title("FA Projection ")
plt.xlabel("F1")
plt.ylabel("F2")
plt.legend(loc='upper left')
plt.savefig(figures_path+os.path.sep+"FA_12"+".png")
plt.close()


plt.figure(figsize=(6, 5))
for i, c,m, label in zip([0,1,2,3,4,5,6,7], 'bgrcmyk2','o+,^vx8s', labels):
    plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 2],
               c=c,marker=m, label=label)

plt.title("FA Projection ")
plt.xlabel("F1")
plt.ylabel("F2")
plt.legend(loc='upper left')
plt.savefig(figures_path+os.path.sep+"FA_13"+".png")
plt.close()


plt.figure(figsize=(6, 5))
for i, c,m, label in zip([0,1,2,3,4,5,6,7], 'bgrcmyk2','o+,^vx8s', labels):
    plt.scatter(X_reduced[y == i, 1], X_reduced[y == i, 2],
               c=c,marker=m, label=label)

plt.title("FA Projection ")
plt.xlabel("F1")
plt.ylabel("F2")
plt.legend(loc='upper left')
plt.savefig(figures_path+os.path.sep+"FA_23"+".png")
plt.close()


kpca = KernelPCA(n_components	=	3,	kernel="rbf",  gamma=0.01)
X_kpca = kpca.fit_transform(X_scaled)



plt.figure(figsize=(6, 5))
for i, c,m, label in zip([0,1,2,3,4,5,6,7], 'bgrcmyk2','o+,^vx8s', labels):
    plt.scatter(X_kpca[y == i, 0], X_kpca[y == i, 1],
               c=c,marker=m, label=label)

plt.title("KPCA Projection  ")
plt.xlabel("1st principal component")
plt.ylabel("2nd principal component")

plt.legend(loc='upper left')
plt.savefig(figures_path+os.path.sep+"KPCAs_12"+".png")
plt.close()

plt.figure(figsize=(6, 5))
for i, c,m, label in zip([0,1,2,3,4,5,6,7], 'bgrcmyk2','o+,^vx8s', labels):
    plt.scatter(X_kpca[y == i, 0], X_kpca[y == i, 2],
               c=c,marker=m, label=label)

plt.title("KPCA Projection  ")
plt.xlabel("1st principal component")
plt.ylabel("2nd principal component")

plt.legend(loc='upper left')
plt.savefig(figures_path+os.path.sep+"KPCAs_13"+".png")
plt.close()


plt.figure(figsize=(6, 5))
for i, c,m, label in zip([0,1,2,3,4,5,6,7], 'bgrcmyk2','o+,^vx8s', labels):
    plt.scatter(X_kpca[y == i, 1], X_kpca[y == i, 2],
               c=c,marker=m, label=label)

plt.title("KPCA Projection  ")
plt.xlabel("1st principal component")
plt.ylabel("2nd principal component")

plt.legend(loc='upper left')
plt.savefig(figures_path+os.path.sep+"KPCAs_23"+".png")
plt.close()