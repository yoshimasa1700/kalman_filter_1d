import numpy as np
import matplotlib.pyplot as plt


class SampleSystem():
    A = 1
    b = 1
    c = 1

    Q = 1  # Variance of system noise.
    R = 10  # Variance of observation noise.

    def __init__(self, N):
        v = np.random.normal(scale=self.Q, size=N)
        w = np.random.normal(scale=self.R, size=N)

        x = np.zeros((N))
        y = np.zeros((N))

        y[0] = self.c * x[0] + w[0]

        for k in range(1, N):
            x[k] = self.A * x[k - 1] + self.b * v[k - 1]
            y[k] = self.c * x[k] + w[k]

        self.x = x
        self.y = y


class KalmanFilter():
    A = 1
    b = 1
    c = 1

    Q = 1  # Variance of system noise.
    R = 10  # Variance of observation noise.

    def __init__(self):
        self.x_hat = 0.0  # Estimated system status.
        self.p = 0.0  # Covariance matrix.

    def run(self, y):
        # Calc a priori estimate
        x_hat_pri = self.A * self.x_hat + self.b * self.Q
        p_pri = self.A * self.p * self.A + self.b * self.Q * self.b

        # Calc kalman gain.
        g = (p_pri * self.c) / (self.c * p_pri * self.c + self.R)

        # Calc filtered result.
        self.x_hat = x_hat_pri + g * (y - self.c * x_hat_pri)
        self.p = (1 - g * self.c) * p_pri

        return self.x_hat


def main():
    np.random.seed(0)

    N = 400

    # define parameter for sample signals.
    ss = SampleSystem(N)

    # define parameter for kalman filter.
    kf = KalmanFilter()

    # input sample signals to kalman filter and collect reuslts.
    x_hat = np.zeros(N)

    for k in range(N):
        x_hat[k] = kf.run(ss.y[k])

    # plot results.
    fig = plt.figure(figsize=(20, 10))

    ax = fig.add_subplot(111)
    ax.plot(ss.x, label="ground truth x")
    ax.plot(ss.y, label="observation y")

    ax.plot(x_hat, label="estimated x")
    ax.grid()

    ax.set_xlabel("time")
    ax.set_ylabel("signal")

    ax.legend()

    plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
