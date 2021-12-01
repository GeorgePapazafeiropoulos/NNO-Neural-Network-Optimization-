function xoverKids  = crossoverFun(parents,options,GenomeLength,~,~,thisPopulation,ratio)
%
% Weighted average crossover
%
% Syntax
%     XOVERKIDS = crossoverFun(PARENTS,OPTIONS,GENOMELENGTH, ...
%         FITNESSFCN,UNUSED,THISPOPULATION,RATIO)
%
% Description
%     Create the crossover children XOVERKIDS of the given population
%     THISPOPULATION using the available PARENTS. Depending on the value of
%     the variable RATIO, children are generated on the line between the
%     parents (if RATIO is scalar) or children are generated within the
%     hypercube with the parents at opposite corners (if RATIO is vector
%     with size [1 x GENOMELENGTH]).
%
% Input arguments
%     PARENTS [double(1 x :inf)] is the vector of parents chosen by the
%         selection function.
%     OPTIONS [struct(1 x 1)] is a structure containing the ga options, 
%         given by the command >>OPTIONS = optimoptions('ga').
%     GENOMELENGTH [double(1 x 1)] is the number of the design variables
%     THISPOPULATION [double(PSIZE x GENOMELENGTH)] contains the
%         individuals in the current population.
%     RATIO [double(1 x 1)] is the weight applied for the weighted average
%         of the parents.
%
% Output arguments
%     XOVERKIDS [double(length(PARENTS)/2 x GENOMELENGTH)] is the
%         offspring that results from the crossover operation.
%
% Example
%     % Create an options structure using crossoverFun as the crossover
%     % function with default ratio = ones(1,GenomeLength)
%     options = optimoptions('ga', 'CrossoverFcn', @crossoverFun);
%     % Create an options structure using crossoverFun as the crossover
%     % function with RATIO of 0.5
%     ratio = 0.5
%     options = optimoptions('ga', 'CrossoverFcn', {@crossoverFun, ratio});
%
% Copyright (c) 2021 by George Papazafeiropoulos
% _________________________________________________________________________
%

% Set default options
if nargin < 7 || isempty(ratio)
    ratio = ones(1,GenomeLength);
end

% How many children to produce?
nKids = length(parents)/2;
% Extract information about linear constraints, if any
linCon = options.LinearConstr;
constr = ~isequal(linCon.type,'unconstrained');
% Allocate space for the kids
xoverKids = zeros(nKids,GenomeLength);

% To move through the parents twice as fast as the kids are being produced,
% a separate index for the parents is needed
index = 1;

for i=1:nKids
    % get parents
    parent1 = thisPopulation(parents(index),:);
    index = index + 1;
    parent2 = thisPopulation(parents(index),:);
    index = index + 1;
    
    % a random number (or vector) on the interval [0,ratio]
    scale = ratio .* rand(1,length(ratio));
    
    % a child gets half from each parent
    xoverKids(i,:) = parent1 + scale .* (parent2 - parent1);
    % Make sure that offspring are feasible w.r.t. linear constraints
    if constr
        
        candKids=xoverKids(i,:)';
        feasible = all((candKids>linCon.lb) & (candKids<linCon.ub));
        
        if ~feasible % Kid is not feasible
            % Children are arithmetic mean of two parents (feasible w.r.t
            % linear constraints)
            alpha = rand;
            xoverKids(i,:) = alpha*parent1 + (1-alpha)*parent2;
        end
    end
end