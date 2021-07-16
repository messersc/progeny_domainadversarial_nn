from sklearn.model_selection import train_test_split


def my_train_test_split(dataset, target, test_size=None):
    if test_size is not None:
        test_size = 0.2
    x_train, x_test, y_train, y_test = train_test_split(
        dataset,
        target,
        test_size=test_size,
        random_state=42,
        stratify=target,
        shuffle=True,
    )
    return(x_train, y_train, x_test, y_test)
