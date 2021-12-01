# NNO-Neural-Network-Optimization-
Neural Network Optimization (NNO) algorithm


Neural Network Optimization (NNO) algorithm for solving nonlinear least-squares (nonlinear data-fitting) problems. The NNO algorithm uses an Artificial Neural Network (ANN) coupled with an arbitrary optimization function, e.g. Genetic Algorithm (GA), towards minimizing the sum of squares of a vector-valued objective function OBJFUN. The ANN is used as a virtual internal objective function equivalent to OBJFUN. The GA algorithm is used for minimizing the ANN. The optimum solution of the ANN given by the GA will be the optimum solution of OBJFUN, since the ANN and the OBJFUN are equivalent. 
It is shown in an example that the number of objective function evaluations required for the NNO algorithm is less than a half of the number of objective function evaluations required for the corresponding Matlab function lsqnonlin. This leads to substantial computational savings, especially if the objective function involves the execution of third-party software (e.g. for a FEA simulation).
The optimization procedure goes as follows: 
(1) An initial set of training data is produced based on OBJFUN 
(2) The ANN is trained based on the above data set. 
(3) The ANN is used as an objective function in GA and is minimized.
(4) OBJFUN is evaluated at the optimum solution that is found by GA.
(5) This extra data is added at the initial set of training data, thus extending the data by one additional OBJFUN function evaluation.
(6) Replace the initial training data with the extended training data
(7) Continue with step (2) above
The NNO algorithm that is implemented in this code submission has already  been successfully used for purposes of constitutive model calibration  involving Finite Element simulation by third-party software (Abaqus) in  the literature:
[1] Papazafeiropoulos, G., Miguel Muñiz Calvente and Emilio Martínez Pañeda , 2017. Abaqus2Matlab: A suitable tool for finite element  post-processing. Advances in Engineering Software, 105, pp.9-16,  doi:10.1016/j.advengsoft.2017.01.006.
[2] Qudama Albujasim and George Papazafeiropoulos. “A Neural Network Inverse Optimization  Procedure for Constitutive Parameter Identification and Failure Mode  Estimation of Laterally Loaded Unreinforced Masonry Walls.” CivilEng,  vol. 2, no. 4, MDPI AG, Nov. 2021, pp. 943–68,  doi:10.3390/civileng2040051.
