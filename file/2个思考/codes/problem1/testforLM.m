clear
clc
close all

epsilon=1e-6;
xdata=[0 1 2 3 4 5];
ydata=[5.2 4.5 2.7 2.5 2.1 1.9];

%LM1
theta0lm=[0,0]';
[thetalm,vallm,klm,normvalue1]=...
    LM1('Fk','JFk',theta0lm,epsilon,xdata,ydata);

%LM2
theta0lm=[0,0]';
[thetalm2,klm2,vallm2,normvalue2]= LM2(epsilon,theta0lm,xdata,ydata);


%Matlab package lsqcurvefit's LM
fun = @(theta,xdata)theta(1)*exp(theta(2)*xdata);
theta0 = [0,0];
%[x,resnorm,residual,exitflag,output] = lsqcurvefit(fun,x0,xdata,ydata);
options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt');
lb = [];
ub = [];
[theta] = lsqcurvefit(fun,theta0,xdata,ydata,lb,ub,options);

%%%%%%%%%

tu1=1;%1
tu2=1;%2


if tu1==1,
%1
times = linspace(xdata(1),xdata(end));
hold on
plot(xdata,ydata,'O-','linewidth',3)
plot(times,fun(thetalm,times),'b-','linewidth',3)
plot(times,fun(thetalm2,times),'yO','linewidth',3)
plot(times,fun(theta,times),'rX','linewidth',3)
legend('Data','Fitted exponential by L-M 1',...
    'Fitted exponential by L-M 2',...
    'Fitted exponential by L-M of lsqcurvefit')
xlabel('$x$','Interpreter','latex','FontSize',40);
ylabel('$\hat{f}(x,\theta)$','Interpreter','latex',...
    'FontSize',40);
%title('Data and Fitted Curve')

set(gcf,'position',[0.5,0.5,1800,900]);
set(gca,'linewidth',3,'fontsize',40,'fontname','Times');
print -depsc -r300 plot/LM
end

if tu2==1,
close all
%2
hold on
plot(0:klm,normvalue1,'bO-','linewidth',3)
plot(0:klm2,normvalue2,'rO-','linewidth',3)
legend('Fitted exponential by L-M 1','Fitted exponential by L-M 2')
%title('the function value residual vesus iteration ')
xlabel('Iteration Number','FontSize',40);
ylabel('$\frac{1}{2}\Vert r(\theta)\Vert^2$','Interpreter','latex',...
    'FontSize',40);
set(gcf,'position',[0.5,0.5,1800,900]);
set(gca,'linewidth',3,'fontsize',40,'fontname','Times');
print -depsc -r300 plot/LM_norm
end
