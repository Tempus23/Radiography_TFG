import matplotlib.pyplot as plt
import numpy as np

class2name = ['Normal','Leve','Moderado','Grave','Muy Grave']

def show_img(img, label):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(class2name[label.numpy()])
    plt.show()