clear all;
close all;

N = 2^11; %Number of time steps in the exact process we are simulating

T = 1; %Final time 
dt = T/N; %Time step
mu = 0.2;
sigma = 1;
x0 = 1;

%% Comparison of the strong convergence of these SDE solvers
%The idea is to take the average over K = 1000 sampled paths with
%different timesteps which are dt*2.^[0:z] with z=5 for example,
%to analyse the strong convergence rate for both the schemes in 
%a log-log plot.

K = 1000;
z = 5;
errorEM = zeros(K,z); %each row refers to a sampled path and there are the different time steps
errorM = zeros(K,z); %each row refers to a sampled path and there are the different time steps
errorPW = zeros(K,z); %each row refers to a sampled path and there are the different time steps
errorP = zeros(K,z); %each row refers to a sampled path and there are the different time steps
errorIE = zeros(K,z); %each row refers to a sampled path and there are the different time steps

for r = 1:K
    %Computing the exact solution and fixing a realization of Brownian
    %motion
    dW = sqrt(dt)*randn(1,N);
    W = cumsum(dW);
    xTrue = x0*exp((mu-0.5*sigma^2)*T+sigma*W(end)); %Exact sol at time t
    
    for s = 1:z
        M = N/2^(s-1);
        n = N/M;
        num_dt = dt*n;
        X = x0;

        % Euler–Maruyama Method
        for i = 1:M
            w = sum(dW(n*(i-1)+1:n*i));
            X = X + mu*X*num_dt + sigma*X*w;
        end
        errorEM(r,s) = abs(X-xTrue);
        X=x0;

        % Milstein Method
        for i = 1:M
            w = sum(dW(n*(i-1)+1:n*i));
            X = X + mu*X*num_dt + sigma*X*w + 0.5*sigma^2*X*(w^2-num_dt);
        end
        errorM(r,s) = abs(X-xTrue);
        
        X=x0;
        % Strongorder 1.5 Taylor method (Platen & Wagner)
        for i = 1:M
            v = sqrt(num_dt)*randn(1);
            w = sum(dW(n*(i-1)+1:n*i));
            Z = 0.5*dt*(w+v/sqrt(3));
            X = X + mu*X*num_dt + sigma*X*w + 0.5*sigma^2*X*(w^2-num_dt)+...
                mu*sigma*X*Z + 0.5*(mu^2*X)*num_dt^2 + (mu*sigma*X)*(w*num_dt-Z)+...
                0.5*sigma^3*X*(w^2/3-num_dt)*w;
        end
        errorPW(r,s) = abs(X-xTrue);
        
        X = x0;
        % Runge-Kutta type scheme of strong order 1
        for i = 1:M
            w = sum(dW(n*(i-1)+1:n*i));
            Xsupp = X + sigma*X*sqrt(num_dt);
            X = X + mu*X*num_dt + sigma*X*w + 0.5/sqrt(num_dt)*sigma*(Xsupp-X)*(w^2-num_dt);
        end
        errorP(r,s) = abs(X-xTrue);
        
        X=x0;
        % Implicit Euler Scheme with strong order 0.5
        for i = 1:M
            w = sum(dW(n*(i-1)+1:n*i));
            X = (X+sigma*X*w)/(1-mu*num_dt);
        end
        errorIE(r,s) = abs(X-xTrue);


    end
    
end

avgEM = mean(errorEM);
avgM = mean(errorM);
avgPW = mean(errorPW);
avgP = mean(errorP);
avgIE = mean(errorIE);

xx = T./(dt*2.^(0:z-1));

ord05 = (xx/xx(end)).^(-0.5)*avgEM(end);
ord1 = (xx/xx(end)).^(-1)*avgEM(end);
ord15 = (xx/xx(end)).^(-1.5)*avgEM(end);

figure;
loglog(xx,avgEM,'-*',xx,ord05,xx,ord1,xx,ord15);
title('Strong convergence rate of Euler-Maruyama');
legend('error','order 0.5','order 1','order 1.5');

ord05 = (xx/xx(end)).^(-0.5)*avgM(end);
ord1 = (xx/xx(end)).^(-1)*avgM(end);
ord15 = (xx/xx(end)).^(-1.5)*avgM(end);

figure;
loglog(xx,avgM,'-*',xx,ord05,xx,ord1,xx,ord15);
title('Strong convergence rate of Milstein');
legend('error','order 0.5','order 1','order 1.5');

ord05 = (xx/xx(end)).^(-0.5)*avgPW(end);
ord1 = (xx/xx(end)).^(-1)*avgPW(end);
ord15 = (xx/xx(end)).^(-1.5)*avgPW(end);

figure;
loglog(xx,avgPW,'-*',xx,ord05,xx,ord1,xx,ord15);
title('Strong convergence rate of Platen & Wagner');
legend('error','order 0.5','order 1','order 1.5');

ord05 = (xx/xx(end)).^(-0.5)*avgP(end);
ord1 = (xx/xx(end)).^(-1)*avgP(end);
ord15 = (xx/xx(end)).^(-1.5)*avgP(end);

figure;
loglog(xx,avgP,'-*',xx,ord05,xx,ord1,xx,ord15);
title('Strong convergence rate of Platen');
legend('error','order 0.5','order 1','order 1.5');

ord05 = (xx/xx(end)).^(-0.5)*avgIE(end);
ord1 = (xx/xx(end)).^(-1)*avgIE(end);
ord15 = (xx/xx(end)).^(-1.5)*avgIE(end);

figure;
loglog(xx,avgIE,'-*',xx,ord05,xx,ord1,xx,ord15);
title('Strong convergence rate of Implicit Euler');
legend('error','order 0.5','order 1','order 1.5');