from mpi4py import MPI
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

# Initialize MPI components
mpi_comm = MPI.COMM_WORLD
current_rank = mpi_comm.Get_rank()
total_processes = mpi_comm.Get_size()

# Data handling on root process
if current_rank == 0:
    print("Loading MNIST dataset...")
    dataset = load_digits()
    X_data, y_labels = dataset.data, dataset.target
    
    # Split dataset into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_labels, test_size=0.2, random_state=42
    )
    
    # Calculate data distribution parameters
    base_chunk = len(X_train) // total_processes
    remainder = len(X_train) % total_processes
else:
    X_train = X_test = y_train = y_test = None
    base_chunk = remainder = None

# Distribute test data and chunk parameters
X_test = mpi_comm.bcast(X_test if current_rank == 0 else None, root=0)
y_test = mpi_comm.bcast(y_test if current_rank == 0 else None, root=0)
base_chunk = mpi_comm.bcast(base_chunk, root=0)
remainder = mpi_comm.bcast(remainder, root=0)

# Prepare data chunks on root process
if current_rank == 0:
    data_chunks = []
    label_chunks = []
    start_index = 0
    
    for proc in range(total_processes):
        chunk_size = base_chunk + (1 if proc < remainder else 0)
        end_index = start_index + chunk_size
        
        data_chunks.append(X_train[start_index:end_index])
        label_chunks.append(y_train[start_index:end_index])
        start_index = end_index
else:
    data_chunks = label_chunks = None

# Distribute data chunks to processes
local_X = mpi_comm.scatter(data_chunks, root=0)
local_y = mpi_comm.scatter(label_chunks, root=0)

print(f"Rank {current_rank}: Training on {len(local_X)} samples")

# Model training and evaluation
train_start = time.time()
model = RandomForestClassifier(n_estimators=10, random_state=current_rank)
model.fit(local_X, local_y)
training_duration = time.time() - train_start

predictions = model.predict(X_test)
local_accuracy = accuracy_score(y_test, predictions)

# Collect results from all processes
combined_preds = mpi_comm.gather(predictions, root=0)
accuracy_data = mpi_comm.gather(local_accuracy, root=0)
time_data = mpi_comm.gather(training_duration, root=0)

# Result aggregation and reporting
if current_rank == 0:
    # Ensemble predictions through majority voting
    vote_matrix = np.zeros((len(y_test), 10), dtype=int)
    for pred_batch in combined_preds:
        for idx, class_pred in enumerate(pred_batch):
            vote_matrix[idx, class_pred] += 1
    
    final_predictions = np.argmax(vote_matrix, axis=1)
    ensemble_accuracy = accuracy_score(y_test, final_predictions)
    
    print("\nResults:")
    print(f"Combined Accuracy: {ensemble_accuracy:.3f}")
    print(f"Mean Training Time: {np.mean(time_data):.3f}s")
    print(f"Total Execution Time: {time.time() - train_start:.3f}s")

