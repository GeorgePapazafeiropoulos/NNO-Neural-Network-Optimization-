function mutationChildren = mutationFun(parents,options,GenomeLength,FitnessFcn,state,thisScore,thisPopulation,mutationRate,mutationScale)
%
% Uniform multi-point mutation
%
% Syntax
%     MUTATIONCHILDREN = MUTATIONFUN(PARENTS,OPTIONS,GENOMELENGTH,...
%         FITNESSFCN,STATE,THISSCORE,THISPOPULATION, ...
%         MUTATIONRATE,MUTATIONSCALE)
%
% Description
%     Create mutated children using uniform mutations at multiple points.
%     Mutated genes are uniformly distributed over the range of the gene.
%     The new value is NOT a function of the parents value for the gene.
%
% Input arguments
%     PARENTS [double(1 x :inf)] is the vector of parents chosen by the
%         selection function.
%     OPTIONS [struct(1 x 1)] is a structure containing the ga options, 
%         given by the command >>OPTIONS = optimoptions('ga').
%     GENOMELENGTH [double(1 x 1)] is the number of the design variables
%     FITNESSFCN [function handle] is the fitness function. It can be
%         called by the command >>Y=FitnessFcn(X), where X [double(N x
%         GENOMELENGTH)] is an array containing N individuals, each
%         containing GENOMELENGTH values of the design variables. Y
%         [double(1 x N)] contains the fitness values of the population X.
%     STATE [struct(1 x 1)] is a structure Structure containing information
%         about the current generation.
%     THISSCORE [double(PSIZE x 1)] contains the scores of the current
%         population. PSIZE [double(1 x 1)] is the population size.
%     THISPOPULATION [double(PSIZE x GENOMELENGTH)] contains the
%         individuals in the current population.
%     MUTATIONRATE [double(1 x 1)] is the mutation rate. Each entry of an
%         individual has a probability rate of being mutated equal to
%         MUTATIONRATE.
%     MUTATIONSCALE [double(1 x 1)] is the mutation scale. If an entry of
%         an individual is being mutated, the new value is given by
%         >>m=lb+MUTATIONSCALE*rand*(ub-lb), where lb, ub are the lower and
%         upper bounds of this entry. If m<lb, then it is set m=lb. If
%         m>ub, then it is set m=ub.
%
% Output arguments
%     MUTATIONCHILDREN [double(length(PARENTS) x GENOMELENGTH)] is the
%         mutated offspring.
%
% Example
%     % Create an options structure specifying that the mutation function
%     % to be used is MUTATIONFUN, with MUTATIONRATE equal to 0.05 and
%     % MUTATIONSCALE equal to 1.2.
%     mutationRate = 0.05;
%     mutationScale = 1.2;
%     options=optimoptions('ga','MutationFcn',...
%           {@mutationFun,mutationRate,mutationScale});
%
% Copyright (c) 2021 by George Papazafeiropoulos
% _________________________________________________________________________
%

% Set default options
if nargin < 9 || isempty(mutationScale)
    mutationScale = 1.2; % default mutation scale
end
if nargin < 8 || isempty(mutationRate)
    mutationRate = 0.01; % default mutation rate
end

if(strcmpi(options.PopulationType,'doubleVector'))
    mutationChildren = zeros(length(parents),GenomeLength);
    for i=1:length(parents)
        child = thisPopulation(parents(i),:);
        % Each element of the genome has mutationRate chance of being mutated.
        mutationPoints = find(rand(1,length(child)) < mutationRate);
        % each gene is replaced with a value chosen randomly from the range.
        range = options.PopInitRange;
        % range can have one column or one for each gene.
        [r,c] = size(range);
        if(c ~= 1)
            range = range(:,mutationPoints);
        end   
        lower = range(1,:);
        upper = range(2,:);
        span = upper - lower;
        
        % Find the child candidate
        childCand=lower + mutationScale*rand(1,length(mutationPoints)) .* span;
        % Correct any lower bound violations
        childCand(childCand<lower)=lower(childCand<lower);
        % Correct any upper bound violations
        childCand(childCand>upper)=upper(childCand>upper);
        
        child(mutationPoints) = childCand;
        
        mutationChildren(i,:) = child;
    end
elseif(strcmpi(options.PopulationType,'bitString'))
    
    mutationChildren = zeros(length(parents),GenomeLength);
    for i=1:length(parents)
        child = thisPopulation(parents(i),:);
        mutationPoints = find(rand(1,length(child)) < mutationRate);
        child(mutationPoints) = ~child(mutationPoints);
        mutationChildren(i,:) = child;
    end
    
end
