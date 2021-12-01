function y = func(x)
%
% Objective function whose sum of squares is minimized
%
% Syntax
%     Y = FUNC(X)
%
% Input arguments
%     X [double(:inf x 1)] is a vector containing the values of the
%         design variables
%
% Output arguments
%     Y [double(:inf x 1)] is a vector containing the values of the
%         function at X.
%
% Example
%     % Evaluate the relative error with respect to the (xI-yI) stress
%     % strain curve for a random combination of design variables
%     x = rand(8,1);
%     y = func(x);
%
% Copyright (c) 2021 by George Papazafeiropoulos
% _________________________________________________________________________
%

% x coordinates of target curve
xI=(0.01:0.01:0.15)';
% y coordinates of target curve
yI=1-100*(xI-0.1).^2;
% Initialize output
YS=zeros(size(yI));
% values of parameters of target curve
param=[0.786755705953708;
    0.475335877910408;
    0.143629528034711;
    0.486456040993623;
    0.906938876165123;
    0.439581830417427;
    0.657547067160447;
    0.460565133380262];
for i=1:15
    k=int64(round(i/2));
    YS(i)=yI(i)*(x(k)-param(k)+1);
end
YT=yI;
y=(YS-YT)./YT;

end

