Mini-HPC and Hybrid HPC-Big Data Clusters Project
Introduction

This project was designed to implement both a traditional High-Performance Computing (HPC) cluster and a hybrid HPC–Big Data cluster using virtual machines.

The main goals were to:

Set up a 3-node cluster (1 master and 2 workers) for distributed machine learning tasks.
Apply distributed ML using MPI and mpi4py.
Build a Spark-based Big Data cluster using Docker Swarm for scalable analytics.
Run and test distributed ML models on datasets such as MNIST and real-world bioinformatics gene expression data.
Methodology

Cluster Setup

Three Ubuntu 24.04 virtual machines were deployed on VirtualBox (1 master node and 2 worker nodes).
Network configuration was completed and passwordless SSH was enabled between all nodes to ensure smooth communication. Connection was verified by accessing worker nodes from the master.
All setup steps and encountered issues were documented during implementation.

Task 1: Mini-HPC Cluster using MPI

OpenMPI and mpi4py were installed on all nodes.
A hostfile was created containing all cluster machines.
Distributed training was executed on the MNIST dataset using a custom Python script (distributed_mnist.py), where both data and computation were split across nodes.

Task 2: Hybrid HPC + Big Data Cluster using Spark

Docker Swarm was initialized across the three virtual machines.
Apache Spark was deployed in cluster mode using a Docker Compose configuration file.
Worker nodes were successfully registered and confirmed via the Spark Web UI.
Distributed machine learning jobs were executed using PySpark on bioinformatics gene expression datasets.
Results

Task 1: Distributed MNIST ML (MPI)

Each process trained on a portion of the MNIST dataset:
Rank 0: 240 samples
Rank 1: 240 samples
Rank 2: 239 samples
Rank 3: 239 samples
Rank 4: 239 samples
Rank 5: 239 samples
Combined Accuracy: 0.942
Mean Training Time: 0.044 seconds
Total Execution Time: 0.073 seconds

Task 2: Spark Cluster (PySpark ML)

Spark cluster was successfully deployed with 2 worker nodes (each with 2 cores and 2GB RAM).
Workers were properly registered and visible in the Spark Web UI, confirming correct setup.
The cluster was fully ready for distributed ML workloads on bioinformatics datasets.
Conclusion

Through this project, we gained practical experience in:

Setting up both traditional HPC and hybrid Big Data clusters using virtual machines.
Implementing distributed machine learning using MPI and Spark frameworks.
Working with real-world bioinformatics datasets in a distributed environment.
Understanding key challenges in distributed computing such as data partitioning and resource management.

Overall, this project provided a solid foundation in distributed systems, HPC, and big data analytics applications.