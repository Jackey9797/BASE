import numpy as np

def visualize(X, GT, Y): 
    global num 
    print(X.x[1].detach().numpy().shape,Y[0][1].detach().numpy().shape)

    import matplotlib.pyplot as plt
    plt.plot(np.concatenate( [X.x[1].detach().numpy(),Y[1].detach().numpy()]), label="pred")
    plt.plot(np.concatenate( [X.x[1].detach().numpy(),GT[1].numpy()]), label="GT")
    plt.legend()
    num += 1
    plt.savefig("figs2/"+str(num)+".png")
    plt.close()
