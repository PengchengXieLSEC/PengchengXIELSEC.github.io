%JFk.m
function JF=JFk(theta) 
xdata=[0 1 2 3 4 5];
ydata=[5.2 4.5 2.7 2.5 2.1 1.9];

JF=[exp(theta(2)*xdata(1)),theta(1)*xdata(1)*exp(theta(2)*xdata(1));
    exp(theta(2)*xdata(2)),theta(1)*xdata(2)*exp(theta(2)*xdata(2));
    exp(theta(2)*xdata(3)),theta(1)*xdata(3)*exp(theta(2)*xdata(3));
    exp(theta(2)*xdata(4)),theta(1)*xdata(4)*exp(theta(2)*xdata(4));
    exp(theta(2)*xdata(5)),theta(1)*xdata(5)*exp(theta(2)*xdata(5));
    exp(theta(2)*xdata(6)),theta(1)*xdata(6)*exp(theta(2)*xdata(6))
    ];

