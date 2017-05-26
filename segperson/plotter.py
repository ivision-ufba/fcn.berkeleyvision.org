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
    plt.clf()
    
    #plot loss
    plt.plot(idx, loss, 'b-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.axis([1, 75, 0, np.max(loss)])
    plt.savefig('loss.png')
