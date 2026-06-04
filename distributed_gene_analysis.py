from mpi4py import MPI
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    # Read data with genes as rows, samples as columns
    df = pd.read_csv("leukemia_expression.csv")
    
    # Transpose to get samples as rows, genes as columns
    df = df.set_index(df.columns[0]).T.reset_index()
    
    # Convert to numerical data and extract target (assuming last column is target)
    data = df.iloc[:, 1:-1].values.astype(float)
    target = df.iloc[:, -1].values
else:
    data = None
    target = None

# Broadcast data to all nodes
data = comm.bcast(data, root=0)
target = comm.bcast(target, root=0)

# Split data horizontally (by samples)
chunk_size = len(data) // size
start = rank * chunk_size
end = (rank + 1) * chunk_size if rank != size - 1 else len(data)

local_data = data[start:end]
local_target = target[start:end]

# Train local model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(local_data, local_target)

# Gather scores
local_score = clf.score(local_data, local_target)
scores = comm.gather(local_score, root=0)

if rank == 0:
    print(f"Classification scores from {size} nodes: {scores}")
    print(f"Average classification accuracy: {np.mean(scores):.4f}")
