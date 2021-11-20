import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(fname="C:/Users/yasmi/Desktop/middle/Breast_cancer_data.txt")

X = data[:, :9]
T = data[:, 9]
X = X.astype(int)
T = T.astype(int)
X_n = data.shape[0]  # 683

# 데이터 분리
X_test = X[:int(X_n / 5 + 3)]
T_test = T[:int(X_n / 5 + 3)]
X_train = X[int(X_n / 5 + 3):]
T_train = T[int(X_n / 5 + 3):]


def gauss(x, mu, s):
    return np.exp(-(x - mu) ** 2 / (2 * s ** 2))


def gauss_func(w, x):
    m = len(w) - 1
    mu = np.linspace(1, 9, m)  # 입력 1~9
    s = mu[1] - mu[0]
    y = np.zeros_like(x)
    for j in range(m):
        y = y + w[j] * gauss(x, mu[j], s)
    y = y + w[m]
    return y


def mse_gauss_func(x, t, w):
    m = len(w) - 1
    y = gauss_func(w, x)
    for j in range(m):
        mse = np.mean((y[:, j] - t) ** 2)
    return mse


def fit_gauss_func(x, t, m):
    mu = np.linspace(1, 9, m)
    s = mu[1] - mu[0]
    n = x.shape[0]
    psi = np.ones((n, m + 1))
    for j in range(m):
        psi = gauss(x, mu[j], s)
    psi_T = np.transpose(psi)

    b = np.linalg.inv(psi_T.dot(psi))
    c = b.dot(psi_T)
    w = c.dot(t)
    return w


def show_gauss_func(w):
    xb = np.linspace(1, 9, 8)
    y = gauss_func(w, xb)
    plt.plot(xb, y, 'r-', linewidth=3)
    plt.xlabel('condition, X')
    plt.ylabel('binary, T')


'''plt.figure(figsize=(8, 8))
M = [3, 4, 6, 9]
for i in range(len(M)):
    plt.subplot(2, 2, i+1)
    W = fit_gauss_func(X,T,M[i])
    show_gauss_func(W)
    plt.plot(X, T, 'bo')
    plt.xlim(1, 9)
    plt.grid(True)
    plt.ylim(0, 1)
    mse = mse_gauss_func(X, T, W)
    plt.title("M={0:d}, SD={1:.1f}".format(M[i], np.sqrt(mse)))
plt.show()'''

T2 = np.zeros((X_n, 9), dtype=np.uint8)

T2[:, 0] = np.abs(T - 1)  # y(t)일때 0 로지스틱에서는 T=1 일때 기준으로 세움 우리가 사용하는 모델은 반대라서 반전


def logistic(x0, x1, x2, x3, x4, x5, x6, x7, x8, w):
    y = 1 / (1 + np.exp(-(
                w[0] * x0 + w[1] * x1 + w[2] * x2 + w[3] * x3 + w[4] * x4 + w[5] * x5 + w[6] * x6 + w[7] * x7 + w[
            8] * x8 + w[9])))
    return y


def cee_logistic(w, x, t):  # 교차 엔트로피 오차 계산
    X_n = x.shape[0]
    y = logistic(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5], x[:, 6], x[:, 7], x[:, 8], w)
    cee = 0
    for n in range(len(y)):
        cee = cee - (t[n, 0] * np.log(y[n]) + (1 - t[n, 0]) * np.log(1 - y[n]))
    cee = cee / X_n
    return cee


'''W= [-1,-1,1,1,-1,1,1,1,-1,1]
print(cee_logistic(W, X, T2)) #5.145'''


def dcee_logistic(w, x, t):
    X_n = x.shape[0]
    y = logistic(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5], x[:, 6], x[:, 7], x[:, 8], w)
    dcee = np.zeros(10)
    for n in range(len(y)):
        dcee[0] = dcee[0] + (y[n] - t[n, :0]) * x[n, 0]
        dcee[1] = dcee[1] + (y[n] - t[n, :0]) * x[n, 1]
        dcee[2] = dcee[2] + (y[n] - t[n, :0]) * x[n, 2]
        dcee[3] = dcee[3] + (y[n] - t[n, :0]) * x[n, 3]
        dcee[4] = dcee[4] + (y[n] - t[n, :0]) * x[n, 4]
        dcee[5] = dcee[5] + (y[n] - t[n, :0]) * x[n, 5]
        dcee[6] = dcee[6] + (y[n] - t[n, :0]) * x[n, 6]
        dcee[7] = dcee[7] + (y[n] - t[n, :0]) * x[n, 7]
        dcee[8] = dcee[8] + (y[n] - t[n, :0]) * x[n, 8]
        dcee[9] = dcee[9] + (y[n] - t[n, :0])
    dcee = dcee / X_n
    return dcee


'''W= [-1,-1,1,1,-1,1,1,1,-1,1]
print(dcee_logistic(W, X, T2))'''  # [3.14, 2.55, 이런거 10개 나옴]

from scipy.optimize import minimize


def fit_logistic(w_init, x, t):  # 최적 w만듬
    res = minimize(cee_logistic, w_init, args=(x, t), jac=dcee_logistic, method="CG")
    return res.x  # 구조체형식으로 최적화된 결과값 x안에 결과값 들어가있음


'''W_init = [1, 1, 0, 1, 0, 1, 0, 1, 0, 0]
W = fit_logistic(W_init, X, T2)
print("w0 = {0:.2f}, w1 = {1:.2f}, w2 = {2:.2f}, w3 = {3:.2f}, w4 = {4:.2f}, w5 = {5:.2f}, w6 = {6:.2f}, w7 = {7:.2f}, w8 ={8:.2f}, w9 ={9:.2f}".format(W[0], W[1], W[2], W[3], W[4], W[5], W[6], W[7], W[8], W[9]))

cee = cee_logistic(W, X, T2)
print("CEE = {0:.2f}".format(cee))
#print("Boundary={0:.2f}".format(B))'''


def kfold_gauss_func(x, t, m, k):
    n = x.shape[0]
    mse_train = np.zeros(k)
    mse_test = np.zeros(k)

    for i in range(0, k):
        x_train = x[np.fmod(range(n), k) != i]
        t_train = t[np.fmod(range(n), k) != i]
        x_test = x[np.fmod(range(n), k) == i]
        t_test = t[np.fmod(range(n), k) == i]
        wm = fit_gauss_func(x_train, t_train, m)
        mse_train[i] = mse_gauss_func(x_train, t_train, wm)
        mse_test[i] = mse_gauss_func(x_test, t_test, wm)
    return mse_train, mse_test


def validate_model(w):
    d = np.loadtxt(fname="C:/Users/yasmi/Desktop/middle/Breast_cancer_data.txt")
    X = d[:, :9].astype(int)
    T = d[:, 9].astype(int)

    N = X.shape[0]

    y = np.zeros(N)
    decision = np.zeros(N).astype(int)
    err_cnt = 0

    print('No.  V   T')
    print('-------------------')
    for i in range(N):
        x = np.r_[X[i, :], 1]
        u = np.array(w).dot(x)
        y[i] = 1 / (1 + np.exp(-u))
        if y[i] < 0.5:
            decision[i] = 1

        if decision[i] != T[i]:
            err_cnt = err_cnt + 1

        print('{0} \t {1} \t {2}'.format(i, decision[i], T[i]))

    hit_ratio = np.round((1 - err_cnt / N) * 100, 1)

    print('---------------------')
    print('Total error : {0} out of {1}'.format(err_cnt, N))
    print('Hit ratio : {0:.1f} %'.format(hit_ratio))

    return hit_ratio


StudentID = "2018146005"
W = np.loadtxt(fname="C:/Users/yasmi/Desktop/middle/2018146005.txt")

print("Student ID: " + StudentID)
print("W " + np.str(W))
print("Wn")

validate_model(W)

print(X_n)
