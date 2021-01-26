import matplotlib.pyplot as plt
from transformer.model import *

history = model.fit([index_inputs, index_outputs], index_targets,
                    batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_split=VALID_SPLIT, callbacks=[earlystop_callback, cp_callback])

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
plt.show()