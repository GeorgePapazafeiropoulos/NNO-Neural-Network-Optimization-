function [xSim,ySim,errSim,errANN,ind,nEval,exitFlag] = NNO(objFun,nVar,lb,ub,initSim,hiddenSizes,Psize,funTol,maxSim,XTol,YTol)
%
% Neural Network Optimization (NNO)
%
% Syntax
%     [XSIM,YSIM,ERRSIM,ERRANN,IND,NEVAL,EXITFLAG] = ...
%         NNO(OBJFUN,NVAR,LB,UB,...
%         INITSIM,hiddenSizes,PSIZE,...
%         FUNTOL,MAXSIM,XTOL,YTOL)
%
% Description
%     Apply the Neural Network Optimization (NNO) algorithm to solve
%     nonlinear least-squares (nonlinear data-fitting) problems. The NNO
%     algorithm uses an Artificial Neural Network (ANN) coupled with a
%     Genetic Algorithm (GA) towards minimizing the sum of squares of a
%     vector-valued objective function. The ANN is used as a dummy internal
%     objective function equivalent to OBJFUN. The GA algorithm is used for
%     minimizing the ANN. The optimum solution of the ANN given by the GA
%     will be the optimum solution of OBJFUN, since the ANN and the OBJFUN
%     are equivalent.
%     The optimization procedure goes as follows:
%     (1) An initial set of training data is produced based on OBJFUN
%     (2) The ANN is trained based on the above data set.
%     (3) The ANN is used as an objective function in GA and is minimized.
%     (4) OBJFUN is evaluated at the optimum solution that is found by GA.
%     (5) This extra data is added at the initial set of training data,
%         thus extending the data by one additional OBJFUN function
%         evaluation.
%     (6) Replace the initial training data with the extended training data
%     (7) Continue with step (2) above
%
% Input arguments
%     OBJFUN [char(1 x :inf)] is the name of the objective function whose
%         sum of squares is minimized. See the file func.m for details
%         about its syntax and a coding example.
%     NVAR [double(1 x 1)] is the number of design variables.
%     LB [double(:inf x 1)] is a vector containing the lower bounds of the
%         design variables
%     UB [double(:inf x 1)] is a vector containing the upper bounds of the
%         design variables
%     INITSIM [double(1 x 1)] is the number of the initial evaluations of
%         OBJFUN before the first training of the ANN.
%     HIDDENSIZES [double(1 x :inf)] is the size of the hidden layers
%         in the ANN, specified as a row vector. The length of the vector
%         determines the number of hidden layers in the ANN.
%     PSIZE [double(1 x 1)] is the size of the population used by the GA.
%     FUNTOL [double(1 x 1)] is the termination tolerance of the objective
%         function. If the sum of squares of OBJFUN becomes lower than
%         FUNTOL, optimum solution is considered to have been reached and
%         the optimization algorithm is terminated.
%     MAXSIM [double(1 x 1)] is the number of maximum OBJFUN function
%         evaluations (NEVAL). If NEVAL>MAXSIM, the optimization algorithm
%         is terminated.
%     XTOL [double(1 x 1)] is the tolerance for the change in the design
%         variables (X). If norm((X(N+1)-X(N))./(abs(X(N+1))+abs(X(N)))) <
%         XTOL, the optimization algorithm is terminated.
%     YTOL [double(1 x 1)] is the tolerance for the change in the output of
%         OBJFUN. If norm((Y(N+1)-Y(N))./(abs(Y(N+1))+abs(Y(N)))) < YTOL,
%         the optimization algorithm is terminated.
%
% Output arguments
%     XSIM [double(NVAR x MAXSIM+1)] contains the values of the design
%         variables that are used throughout the whole optimization
%         history. The optimum values of the NNO are equal to
%         XSIM(:,IND(1)).
%     YSIM [double(:inf x MAXSIM+1)] contains the values of OBJFUN that are
%         used throughout the whole optimization history. The optimized
%         output of OBJFUN is equal to YSIM(:,IND(1)).
%     ERRSIM [double(1 x MAXSIM+1)] contains the error which is equal to
%         the sum of squares of OBJFUN throughout the whole optimization
%         history. The optimized error of OBJFUN is equal to
%         ERRSIM(IND(1)).
%     ERRANN [double(1 x MAXSIM+1)] contains the error which is equal to
%         the sum of squares of the output of the ANN, used as a dummy
%         objective function equivalent to OBJFUN, throughout the whole
%         optimization history. The optimized error of ANN is equal to
%         ERRANN(IND(1)).
%     IND [double(1 x 1)] is the position of the optimum in the
%         optimization history.
%     NEVAL [double(1 x 1)] is the number of OBJFUN function evaluations.
%     EXITFLAG [double(1 x 1)] is an integer, showing the reason the solver
%         stopped. It can take the following values (compatible with the
%         exitflag output of the Matlab function LSQNONLIN):
%         EXITFLAG=1: Function converged to a solution
%         EXITFLAG=2: Change in X is less than the specified tolerance TOLX
%         EXITFLAG=3: Change in Y is less than the specified tolerance TOLY
%         EXITFLAG=0: Number of function evaluations exceeded MAXSIM
%         EXITFLAG=-1: An error in OBJFUN stopped the solver.
%
% Example
%     objFun='func';
%     nVar=8;
%     lb=0*ones(nVar,1);
%     ub=1*ones(nVar,1);
%     initSim=5;
%     hiddenSizes = 15; % row vector
%     Psize=10;
%     funTol=0.001;
%     maxSim=60;
%     XTol=0.001;
%     YTol=0.001;
%     [xSim,ySim,errSim,errANN,ind,nEval,exitFlag] = ...
%         NNO(objFun,nVar,lb,ub,... % optimization properties
%         initSim,hiddenSizes,Psize,... % ANN/GA properties
%         funTol,maxSim,XTol,YTol); % termination properties
%     % Output of the Neural Network Optimization algorithm
%     % Evolution of the OBJFUN error
%     figure(1)
%     plot(log(errSim))
%     xlabel('Iteration')
%     ylabel('log(error)')
%     title('OBJFUN error')
%     % Evolution of the optimum point of the dummy ANN function
%     figure(2)
%     plot(log(errANN(initSim+1:end)))
%     xlabel('Iteration')
%     ylabel('log(error)')
%     title('ANN error')
%     % Print the optimum values of the design variables
%     xSim(:,ind(1))
%     % Compare the target curve and the optimum curve
%     % x coordinates of target curve
%     xI=(0.01:0.01:0.15)';
%     % y coordinates of target curve
%     yI=1-100*(xI-0.1).^2;
%     % Optimum curve based on the optimum values of the design variables
%     yOpt = func(xSim(:,ind(1)));
%     yOpt=sqrt(yOpt).*yI+yI;
%     % Plot
%     figure(3)
%     plot(xI,yI,'Color','black')
%     hold on
%     plot(xI,yOpt,'Color','red')
%     hold off
%
% Copyright (c) 2021 by George Papazafeiropoulos
% _________________________________________________________________________
%

%% STEP 1: Input checks
if nargin<11
    error('Missing variable definitions');
end
% Checks for objFun
if ~isa(objFun,'char')
    error('objFun must be a row character array');
end
if isempty(objFun)
    error('objFun is empty');
end
if ~isempty(objFun) && size(objFun,1)~=1
    error('objFun must be a row vector');
end
if exist([objFun,'.m'], 'file') == 2
else
    error('objFun does not exist');
end
% Checks for nVar
if ~isa(nVar,'double')
    error('nVar must be double');
end
if isempty(nVar)
    error('nVar is empty');
end
if any(~isfinite(nVar))
    error('Inf or NaN values in nVar');
end
if ~isempty(nVar) && numel(nVar)~=1
    error('nVar must be scalar');
end
if (floor(nVar)~=nVar) || (nVar<1)
    error('nVar must be a positive integer');
end
% Checks for lb
if ~isa(lb,'double')
    error('lb must be double');
end
if isempty(lb)
    error('lb is empty');
end
if any(~isfinite(lb))
    error('Inf or NaN values in lb');
end
if ~isempty(lb) && size(lb,2)~=1
    error('lb must be column vector');
end
if size(lb,1)~=nVar
    error('lb must contain nVar elements');
end
% Checks for ub
if ~isa(ub,'double')
    error('ub must be double');
end
if isempty(ub)
    error('ub is empty');
end
if any(~isfinite(ub))
    error('Inf or NaN values in ub');
end
if ~isempty(ub) && size(ub,2)~=1
    error('ub must be column vector');
end
if size(ub,1)~=nVar
    error('ub must contain nVar elements');
end
% Checks for initSim
if ~isa(initSim,'double')
    error('initSim must be double');
end
if isempty(initSim)
    error('initSim is empty');
end
if any(~isfinite(initSim))
    error('Inf or NaN values in initSim');
end
if ~isempty(initSim) && numel(initSim)~=1
    error('initSim must be scalar');
end
if (floor(initSim)~=initSim) || (initSim<1)
    error('initSim must be a positive integer');
end
% Checks for hiddenSizes
if ~isa(hiddenSizes,'double')
    error('hiddenSizes must be double');
end
if isempty(hiddenSizes)
    error('hiddenSizes is empty');
end
if any(~isfinite(hiddenSizes))
    error('Inf or NaN values in hiddenSizes');
end
if ~isempty(hiddenSizes) && size(hiddenSizes,1)~=1
    error('hiddenSizes must be row vector');
end
if any(floor(hiddenSizes)~=hiddenSizes) || any(hiddenSizes<1)
    error('hiddenSizes must contain positive integers');
end
% Checks for Psize
if ~isa(Psize,'double')
    error('Psize must be double');
end
if isempty(Psize)
    error('Psize is empty');
end
if any(~isfinite(Psize))
    error('Inf or NaN values in Psize');
end
if ~isempty(Psize) && numel(Psize)~=1
    error('Psize must be scalar');
end
if (floor(Psize)~=Psize) || (Psize<1)
    error('Psize must be a positive integer');
end
% Checks for funTol
if ~isa(funTol,'double')
    error('funTol must be double');
end
if isempty(funTol)
    error('funTol is empty');
end
if any(~isfinite(funTol))
    error('Inf or NaN values in funTol');
end
if ~isempty(funTol) && numel(funTol)~=1
    error('funTol must be scalar');
end
if funTol<0
    error('funTol must be positive');
end
% Checks for XTol
if ~isa(XTol,'double')
    error('XTol must be double');
end
if isempty(XTol)
    error('XTol is empty');
end
if any(~isfinite(XTol))
    error('Inf or NaN values in XTol');
end
if ~isempty(XTol) && numel(XTol)~=1
    error('XTol must be scalar');
end
if XTol<0
    error('XTol must be positive');
end
% Checks for YTol
if ~isa(YTol,'double')
    error('YTol must be double');
end
if isempty(YTol)
    error('YTol is empty');
end
if any(~isfinite(YTol))
    error('Inf or NaN values in YTol');
end
if ~isempty(YTol) && numel(YTol)~=1
    error('YTol must be scalar');
end
if YTol<0
    error('YTol must be positive');
end
% Checks for maxSim
if ~isa(maxSim,'double')
    error('maxSim must be double');
end
if isempty(maxSim)
    error('maxSim is empty');
end
if any(~isfinite(maxSim))
    error('Inf or NaN values in maxSim');
end
if ~isempty(maxSim) && numel(maxSim)~=1
    error('maxSim must be scalar');
end
if (floor(maxSim)~=maxSim) || (maxSim<1)
    error('maxSim must be a positive integer');
end
if maxSim<initSim
    error('maxSim must not be lower than initSim')
end

%% STEP 2: Initializations
% Number of objective function evaluations
nEval=0;
% Simulation error
errSim=zeros(1,maxSim+1);
% ANN error
errANN=zeros(1,maxSim+1);
% Optimum index
ind=[];
% Exit flag
exitFlag=1;
% Iteration counter
j=initSim;

%% STEP 3: Define initial points to train the neural net
% uniformly distributed
xSim=[repmat(lb,1,initSim)+repmat(ub-lb,1,initSim).*rand(nVar,initSim),...
    zeros(nVar,maxSim-initSim+1)];
% normally distributed
%p=6;
%xSim=[repmat(lb,1,initSim)+repmat(ub-lb,1,initSim).*sum(rand(nVar,initSim,p),3)/p,...
%    zeros(nVar,maxSim-initSim+1)];
% Define initial population for the ga
P=repmat(lb',Psize,1)+repmat(ub'-lb',Psize,1).*rand(Psize,nVar);

%% STEP 5: Simulate the initial points
% Evaluate residual function and initialize the simulated curves array
% based on the residual function output
try
    A = feval(objFun,xSim(:,1));
    nEval=nEval+1;
catch
    exitFlag=-1;
    return;
end
% Checks for the output of objFun
objFunOutputChk(A)
% Initialize simulated curves
nA=size(A,1);
ySim=zeros(nA,maxSim+1);
ySim(:,1) = A;
% Estimate current error
errSim(:,1)=sum(ySim(:,1).^2);
for sim=2:initSim
    % Evaluate residual function
    try
        A = feval(objFun,xSim(:,sim));
        nEval=nEval+1;
    catch
        exitFlag=-1;
        return;
    end
    % Checks for the output of objFun
    objFunOutputChk(A)
    ySim(:,sim) = A;
    % Estimate current error
    errSim(:,sim)=sum(ySim(:,sim).^2);
end

%% STEP 6: Main engine
% Start iterative procedure
while errSim(j)>funTol
    
    %% STEP 6.1: Train the ANN
    % Choose a Training Function
    trainFcn = 'trainbr';  % Bayesian Regularization
    %trainFcn = 'trainlm';  % Levenberg-Marquardt
    % Create a Fitting Network
    net = fitnet(hiddenSizes,trainFcn);
    % Control number of epochs to avoid overfitting (this renders the ANN
    % not suitable as an objective function for an optimization procedure)
    net.trainParam.epochs=500;
    % Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 5/100;
    % Do not show the ANN training GUI window
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    % Train the Network
    [net,tr] = train(net,xSim(:,1:j),ySim(:,1:j));
    
    %% STEP 6.2: Find optimization starting point
    [~,ind]=sort(errSim(1:j));
    %ind2=ind(1);
    
    %% STEP 6.3: Optimization process using the ANN as objective function
    % Define local objective function
    objFunLoc=@(x) objFunLocNest(x,net);
    % Minimize the local objective function
    options = optimoptions('ga');
    options.Display='off'; %'iter';
    options.PopulationSize=Psize;
    options.InitialPopulationRange=[lb';ub'];
    %options.PopInitRange=[lb';ub'];
    options.InitialPopulationMatrix=P;
    options.CreationFcn=@gacreationuniform;
    %options.CreationFcn=@gacreationlinearfeasible;
    options.MutationFcn={@mutationFun,0.05,1.4};
    r=1.4*ones(1,nVar);
    options.CrossoverFcn={@crossoverFun,r};
    options.CrossoverFraction = 0.9;
    options.MaxGenerations=50;
    %options.MaxStallGenerations=10;
    [x_opt,f_opt,exitflag,output,P]=ga(objFunLoc,nVar,[],[],[],[],lb,ub,[],options);
    
    % Increase iteration counter
    j=j+1;
    % Save optimization results
    xSim(:,j)=x_opt;
    errANN(j)=f_opt;
    
    %% STEP 6.4: Simulate the optimal point
    % Evaluate residual function
    try
        A = feval(objFun,x_opt);
        nEval=nEval+1;
    catch
        exitFlag=-1;
        return;
    end
    % Checks for the output of objFun
    objFunOutputChk(A)
    ySim(:,j) = A;
    % Estimate current error
    errSim(j)=sum(ySim(:,j).^2);
    
    %% STEP 6.5: Apply stopping criteria
    % Check the difference between relative norm of the values of design
    % variables for two successive iterations.
    varX=norm((xSim(:,j)-xSim(:,j-1))./(abs(xSim(:,j))+abs(xSim(:,j-1))));
    % If there is not significant difference, stop the process
    if varX<XTol
        warning('Change in x is less than the specified tolerance')
        exitFlag=2;
        return;
    end
    % Check the difference between relative norm of the values of design
    % variables for two successive iterations.
    varY=norm((ySim(:,j)-ySim(:,j-1))./(abs(ySim(:,j))+abs(ySim(:,j-1))));
    % If there is not significant difference, stop the process
    if varY<YTol
        warning('Change in the residual is less than the specified tolerance')
        exitFlag=3;
        return;
    end
    % If the maximum number of iterations is exceeded, stop the process
    if j>maxSim
        warning('Number of function evaluations exceeded maxSim')
        exitFlag=0;
        return;
    end
    
end

end

function y = objFunLocNest(x,net,varargin)
% Local objective function
% a1=varargin{1};
% etc...
y=net(x');
y=sum(y.^2);
end

function objFunOutputChk(A)
% Checks for the output of objFun
if ~isa(A,'double')
    error('Output of objFun must be double');
end
if isempty(A)
    error('Output of objFun is empty');
end
if any(~isfinite(A))
    error('Inf or NaN values in Output of objFun');
end
if ~isempty(A) && size(A,2)~=1
    error('Output of objFun must be column vector');
end
end