%% Fit a stress strain curve with 8 parameters
% An arbitrary stress strain curve is fitted. The number of required
% objective function evaluations is compared for the NNO algorithm and for
% the conventional least squares algorithm (lsqnonlin).

%% Define input data for the NNO algorithm
% Name of residual function
objFun='func';

%%
% Number of design variables
nVar=8;

%%
% Lower and upper bound vectors
lb=0*ones(nVar,1);
ub=1*ones(nVar,1);

%%
% Number of Abaqus analyses for initial training of the neural network
initSim=5;

%%
% Number and size of hidden layers
hiddenSizes = 15; % row vector

%%
% Population size
Psize=10;

%%
% Termination tolerance of error between target and simulated curve
funTol=0.0005;

%%
% Maximum number of iterations
maxSim=60;

%%
% Stall tolerance for X
XTol=0.001;

%%
% Stall tolerance for Y
YTol=0.001;

%%
% Set rng for repeatability
rng(0)

%% Solution with the Neural Network Optimization algorithm
% Apply the NNO function
[xSim,ySim,errSim,errANN,ind,nEval1,exitFlag] = ...
    NNO(objFun,nVar,lb,ub,... % optimization properties
    initSim,hiddenSizes,Psize,... % ANN/GA properties
    funTol,maxSim,XTol,YTol); % termination properties

%% Output of the Neural Network Optimization algorithm
% Check the evolution of the OBJFUN error
figure(1)
plot(log(errSim))
xlabel('Iteration')
ylabel('log(error)')
title('OBJFUN error')

%%
% Check the evolution of the optimum point of the dummy ANN function
figure(2)
plot(log(errANN(initSim+1:end)))
xlabel('Iteration')
ylabel('log(error)')
title('ANN error')

%%
% Print the optimum values of the design variables
xSim(:,ind(1))

%% Compare the target curve and the optimum curve
% x coordinates of target curve
xI=(0.01:0.01:0.15)';

%%
% y coordinates of target curve
yI=1-100*(xI-0.1).^2;

%%
% Optimum curve based on the optimum values of the design variables
yOpt1 = func(xSim(:,ind(1)));
yOpt1=yOpt1.*yI+yI;

%%
% Plot
figure(3)
plot(xI,yI,'Color','black')
hold on
plot(xI,yOpt1,'Color','red')
hold off
title('Neural Network Optimization')
xlabel('X')
ylabel('Y')

%% Solution with the lsqnonlin function
% Apply the lsqnonlin function
x0=lb+rand(8,1).*(ub-lb);
options=optimset('lsqnonlin');
options.TolFun=0.02;
[x,resnorm,residual,exitflag,output] = lsqnonlin(objFun,x0,lb,ub,options);

%%
% Optimum curve based on the optimum values of the design variables
yOpt2 = func(x);
yOpt2=yOpt2.*yI+yI;

%%
% Plot
figure(4)
plot(xI,yI,'Color','black')
hold on
plot(xI,yOpt2,'Color','red')
hold off
title('lsqnonlin')
xlabel('X')
ylabel('Y')

%% Compare number of objective function evaluations
% For the proposed Neural Network Optimization algorithm:
nEval1

%%
% For the conventional lsqnonlin optimization algorithm:
nEval2=output.funcCount

%%
%
% Copyright (c) 2021 by George Papazafeiropoulos
%
