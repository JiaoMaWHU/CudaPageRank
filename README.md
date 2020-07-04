# PageRankCuda
This is github repository of EE361C term project: "A Parallel GPU Implementation of PageRank Algorithm".

To run our implementation (facilitated version), please follow the instructions below:
+ Make sure you have the access to TACC or other mahcine which enables sbatch.
+ Load `cuda` and `gcc` library.
+ Starting from the home directory of this respository, go to the power folder: `cd power`.
+ Open the main.cu program, choose the dataset you want, which is located in the data folder.
+ Give running priviledge to the script in the power folder using `chmod +x run.sh`.
+ Excute the script: `./run.sh`.
+ Then, you will get the output in the `power/output` folder. 

**Test Criteria**: if the algorithm terminates in a reasonable time, and the end differece is smaller than the requirement (e.g. 1e-4), the algorithm is successful.
