import matplotlib.pyplot as plt
import numpy as np

def visualize(cm, class_names, normalized = True):
    
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    #normalizza la matrice di confusione se normalized è True.
    if normalized:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    threshold = cm.max() / 2.
    
    #setta il colore del testo: bianco se è a sfondo scuro, nero altrimenti
    for i in range(cm.shape[0]):
      for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')