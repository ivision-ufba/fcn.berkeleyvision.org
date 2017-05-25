import numpy as np
import matplotlib.pyplot as plt

def plot(acc, loss, maxepoch=75):
    idx = np.arange(1, len(acc) + 1)
    
    # plot accuracy
    plt.plot(idx, acc, 'r-')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.axis([1, 75, 0, 100])
    plt.savefig('accuracy.png')
    plt.show()
    
    #plot loss
    plt.plot(idx, loss, 'b-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.axis([1, 75, 0, np.max(loss)])
    plt.savefig('loss.png')

if __name__ == '__main__':
    np.random.seed(1337)
    acc = np.random.uniform(0, 100, 75)
    loss = np.random.uniform(0, 1829234, 75)
    plot(acc, loss)
