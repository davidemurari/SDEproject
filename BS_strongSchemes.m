clear all;
close all;

N = 2^8; %Number of time steps in the exact process we are simulating
M = N/8; %Number of time steps in the numerical simulation

T = 1; %Final time 
n = N/M;
dt = T/N; %Time step
num_dt = dt*n;

mu = 0.2;
sigma = 1;
x0 = 1;
valuesEM = zeros(M,1); %Stored values for Euler-Maruyama Method
valuesM = zeros(M,1); %Stored values for Milstein Method
valuesP = zeros(M,1); %Stored values for Platen Method
valuesIE = zeros(M,1); %Stored values for Implicit Euler


%Computing the exact solution and fixing a realization of Brownian
%motion
dW = sqrt(dt)*randn(1,N);
W = cumsum(dW);
xTrue = x0*exp((mu-0.5*sigma^2)*[dt:dt:T]+sigma*W); %Exact sol at time t

X = x0;
% Euler–Maruyama Method
for i = 1:M
    w = sum(dW(n*(i-1)+1:n*i));
    X = X + mu*X*num_dt + sigma*X*w;
    valuesEM(i)=X;
end

X=x0;
% Milstein Method
for i = 1:M
    w = sum(dW(n*(i-1)+1:n*i));
    X = X + mu*X*num_dt + sigma*X*w + 0.5*sigma^2*X*(w^2-num_dt);
    valuesM(i)=X;
end

X=x0;
% Runge-Kutta type scheme of strong order 1
for i = 1:M
    w = sum(dW(n*(i-1)+1:n*i));
    Xsupp = X + mu * X * num_dt + sigma*X*sqrt(num_dt);
    X = X + mu*X*num_dt + sigma*X*w + 0.5/sqrt(num_dt)*sigma*(Xsupp-X)*(w^2-num_dt);
    valuesP(i)=X;
end

X=x0;
% Implicit Euler Scheme with strong order 0.5
for i = 1:M
    w = sum(dW(n*(i-1)+1:n*i));
    X = (X+sigma*X*w)/(1-mu*num_dt);
    valuesIE(i)=X;
end


%% Plots comparing the 2 schemes in terms of visual fitting

figure;
plot([0:num_dt:T],[x0;valuesEM],'r-o');
hold on;
plot([0:dt:T],[x0,xTrue],'g-');
title('Comparison between exact sol and Euler–Maruyama Method');

figure;
plot([0:num_dt:T],[x0;valuesM],'r-o');
hold on;
plot([0:dt:T],[x0,xTrue],'g-');
title('Comparison between exact sol and Milstein Method');

figure;
plot([0:num_dt:T],[x0;valuesP],'r-o');
hold on;
plot([0:dt:T],[x0,xTrue],'g-');
title('Comparison between exact sol and Platen Method');

figure;
plot([0:num_dt:T],[x0;valuesIE],'r-o');
hold on;
plot([0:dt:T],[x0,xTrue],'g-');
title('Comparison between exact sol and Implicit Euler method');