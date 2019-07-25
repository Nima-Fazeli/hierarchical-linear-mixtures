from src.LinMix import MixLinBayes
from example.data_dummy import dummy_data
import numpy as np
import matplotlib.pyplot as plt

def main():
    # init data class
    dd = dummy_data.DummyData()
    # generate the data
    dd.linear_transform(with_shuffle=False)

    x = dd.x
    y = dd.y

    # initialize model with the training data
    # x - NxF numpy array (F is dimension of feature)
    # y - NxD numpy array (D is output dimension)
    lm = MixLinBayes(x, y)
    # train model
    lm.train()

    ## test model
    # pick data range to make predictions over
    pp_x = np.linspace(- 1.05, 1.05, 100)
    # use model to predict
    y_pred_mean = lm.predict(pp_x)

    ## plot prediction resutls
    fig, ax = plt.subplots(figsize=(8, 6))
    # plot original data
    ax.scatter(x, y, c='b', zorder=10, label=None)
    # plot predictions of the model over the data range
    ax.plot(pp_x, y_pred_mean, c='k', zorder=6, label='Posterior expected value')
    # plot formatting
    ax.legend(loc=1)
    ax.set_title('Data')

    plt.show()


if __name__ == '__main__':
    main()
