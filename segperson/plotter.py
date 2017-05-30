import numpy as np
import matplotlib.pyplot as plt

def plot(acc, loss, it, maxepoch=75):
    idx = np.arange(1, it + 2)
    
    # plot accuracy
    plt.plot(idx, acc[:it + 1], 'r-')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.axis([1, maxepoch, 0, 100])
    plt.savefig('accuracy.png')
    plt.clf()
    
    #plot loss
    plt.plot(idx, loss[:it + 1], 'b-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.axis([1, maxepoch, 0, np.max(loss)])
    plt.savefig('loss.png')
