def get_accuracy(model, loader):
    # https://www.cs.toronto.edu/~lczhang/321/tut/tut04.pdf
    correct, total = 0, 0
    for xs, ts in loader:
        zs = model(xs)
        # get the index of the max logit
        pred = zs.max(1, keepdim=True)[1]
        correct += pred.eq(ts.view_as(pred)).sum().item()
        total += int(ts.shape[0])
    return correct / total
