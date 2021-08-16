%Fk.m
function y=Fk(theta) 
xdata=[0 1 2 3 4 5];
ydata=[5.2 4.5 2.7 2.5 2.1 1.9];
y(1)=theta(1)*exp(theta(2)*xdata(1))-ydata(1); 
y(2)=theta(1)*exp(theta(2)*xdata(2))-ydata(2); 
y(3)=theta(1)*exp(theta(2)*xdata(3))-ydata(3); 
y(4)=theta(1)*exp(theta(2)*xdata(4))-ydata(4); 
y(5)=theta(1)*exp(theta(2)*xdata(5))-ydata(5); 
y(6)=theta(1)*exp(theta(2)*xdata(6))-ydata(6); 
y=y(:);

