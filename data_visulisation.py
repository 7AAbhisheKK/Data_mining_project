from matplotlib import pyplot as plt
from data import load_dataset_from_file
def dataset():
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_from_file()
    # print("X_train [shape %s] sample patch:\n" % (str(X_train.shape)), X_train[1, 15:20, 5:10])
    # print("A closeup of a sample patch:")
    # plt.imshow(X_train[1, 15:20, 5:10], cmap="Greys")
    # plt.show()
    # print("And the whole sample:")
    # plt.imshow(X_train[1], cmap="Greys")
    # plt.show()
    # print("y_train [shape %s] 10 samples:\n" % (str(y_train.shape)), y_train[:10])
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    X_val_flat = X_val.reshape((X_val.shape[0], -1))
    X_test_flat=X_test.reshape((X_test.shape[0],-1))
    print(y_train.shape)
    return X_train_flat.T,y_train,X_test_flat.T,y_test
    