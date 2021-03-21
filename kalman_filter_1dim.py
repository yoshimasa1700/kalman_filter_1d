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

        # store kalman gain for study.
        self.g = g

        # Calc filtered result.
        self.x_hat = x_hat_pri + g * (y - self.c * x_hat_pri)
        self.p = (1 - g * self.c) * p_pri

        return self.x_hat


def plot_result(value_and_label_list, image_name,
                xlabel="time", ylabel="signal",
                show=False):
    fig = plt.figure(figsize=(20, 10))

    ax = fig.add_subplot(111)

    for v, l in value_and_label_list:
        ax.plot(v, label=l)

    ax.grid()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend()

    if show:
        plt.show()
    plt.savefig(image_name)

    plt.close(fig)


def main():
    np.random.seed(0)

    N = 400

    # define parameter for sample signals.
    ss = SampleSystem(N)

    # define parameter for kalman filter.
    kf = KalmanFilter()

    # input sample signals to kalman filter and collect reuslts.
    x_hat = np.zeros(N)
    g = np.zeros(N)

    for k in range(N):
        x_hat[k] = kf.run(ss.y[k])
        g[k] = kf.g

    # plot results.
    plot_result(
        [
            (ss.x, "ground truth x"),
            (ss.y, "observation y"),
            (x_hat, "estimated x"),
        ],
        "kalman_filter_1dim.png"
    )

    plot_result(
        [
            (g, "kalman gain")
        ],
        "kalman_gain.png"
    )


if __name__ == "__main__":
    main()
