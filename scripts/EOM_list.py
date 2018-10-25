import numpy as np


listeom = np.linspace(8.5, -8.5, 50)
#listeom = np.ndarray.tolist(listeom)
listeom = ["%.2f"%l for l in listeom]

print ",".join(listeom)

