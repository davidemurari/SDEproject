clear all;
close all;

Cm = 1;
gk = 36;
gNa = 120;
gl = 0.3;
vk = -12;
vNa = 115;
vl = 10.613;
an = @(v) (10-v)/(100*(exp((10-v)/10)-1));
dan = @(v) (-(exp((10-v)/10)-1) - (10-v)*exp((10-v)/10)*(-1/10))/(100*(exp((10-v)/10)-1)^2);
am = @(v) (25-v)/(10*(exp((25-v)/10)-1));
dam = @(v) (-(exp((25-v)/10)-1) - (25-v)*exp((25-v)/10)*(-1/10))/(10*(exp((25-v)/10)-1)^2);
ah = @(v) 7/100 * exp(-v/20);
dah = @(v) -7/100 * 1/20 * exp(-v/20);
bn = @(v) 1/8 * exp(-v/80);
dbn = @(v) -1/8 * 1/80 * exp(-v/80);
bm = @(v) 4*exp(-v/18);
dbm = @(v) -4 * 1/18 * exp(-v/18);
bh = @(v) 1/(1+exp((30-v)/10));
dbh = @(v) -exp((30-v)/10)*(-1/10)/(exp((30-v)/10)+1)^2;

mu = 3; %Determines the applied current

sigmavector = [0,2.5,10]; %Vector with the stochastic parameters to test different things
p = 1; 

for sigma = sigmavector
    b = [sigma;0;0;0]./Cm;
    % y = [v,n,m,h]'
    F = @(y) [(gk*y(2)^4*(vk-y(1))+gNa*y(3)^3*y(4)*(vNa-y(1))+gl*(vl-y(1))+mu)/Cm ;
              an(y(1))*(1-y(2))-bn(y(1))*y(2);
              am(y(1))*(1-y(3))-bm(y(1))*y(3);
              ah(y(1))*(1-y(4))-bh(y(1))*y(4)];
    %Problem to be solved: dy = F(y)* dt+ b * dw(t)
    %To do that we are going to use Implicit Euler because of the stiffness
    %of the problem

    DF = @(y) [(-gk*y(2)^4-gNa*y(3)^3*y(4)-gl)/Cm, (4*gk*y(2)^3*(vk-y(1)))/Cm,...
                (3*gNa*y(3)^2*y(4)*(vNa-y(1)))/Cm, (gNa*y(3)^3*(vNa-y(1)))/Cm; %First row
               dan(y(1))*(1-y(2))-dbn(y(1))*y(2), -an(y(1))-bn(y(1)), 0, 0;
               dam(y(1))*(1-y(3))-dbm(y(1))*y(3), 0, -am(y(1))-bm(y(1)), 0;
               dah(y(1))*(1-y(4))-dbh(y(1))*y(4), 0, 0, -ah(y(1))-bh(y(1))];

    %The implicit Euler scheme writes:
    % Y_{n+1} = Y_{n} + a(t_{n+1},Y_{n+1})dt + b(t_n,Y_n)dW_{t_n}
    %So we need to use for example Euler method to solve:
    % G(x) - Y_n - b(t_n,Y_n)dW_{t_n} = 0
    % in the unknown x = Y_{n+1} and where DG(x) = eye(4)-DF(x)*dt

    v0 = 0;
    n0 = an(v0)/(an(v0)+bn(v0));
    m0 = am(v0)/(am(v0)+bm(v0));
    h0 = ah(v0)/(ah(v0)+bh(v0));

    X0 = [v0;n0;m0;h0]; %Initial condition
    X=X0;

    M = 2^12; %Number of time steps in the numerical simulation

    T = 250; %Final time 
    dt = T/(M-1); %Time step

    valuesIE = zeros(4,M); %Stored values for Implicit Euler

    tol=dt;

    G = @(x,y0) y0 + F(x)*dt - x + b.*sqrt(dt)*randn;
    DG = @(x) dt*DF(x) - eye(length(x));
    % Implicit Euler Scheme with strong order 0.5
    for i = 1:M
        %Newton method
        y=X;
        res=-DG(y)\G(y,X);
        while norm(res)>tol
            y=y+res;
            res=-DG(y)\G(y,X);
        end
        X=y+res;
        valuesIE(:,i)=X;
    end
    
    time = (0:dt:T);
    % plot(time,valuesIE(1,:),'k-',time,100*valuesIE(2,:),'r-',time,100*valuesIE(3,:),'g-',time,100*valuesIE(4,:),'b-');
    % legend('v','n x 100','m x 100','h x 100');    
    string = strcat(texlabel('mu = '),num2str(mu),' , ',texlabel('sigma = '),num2str(sigma));
    subplot(length(sigmavector),2,p);
    plot(time,valuesIE(1,:),'k-');
    title(string);
    xlabel('time');
    ylabel('v');

    p = p + 1; %Just the position in the subplot
    
    subplot(length(sigmavector),2,p);
    plot(valuesIE(1,:),valuesIE(2,:),'r-')
    title(string);
    xlabel('v');
    ylabel('n');
    axis([-50 150 0 1]);
    p = p + 1;
end
