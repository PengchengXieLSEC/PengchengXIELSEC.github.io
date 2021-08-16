function [t,k,val,vallist]= LM2(epsilon,theta0,xdata,ydata)

f_x=@(x,theta1,theta2,y) ( theta1*exp(theta2*x)-y);
J_x=@(x,theta1,theta2) ([ exp(theta2*x),theta1*exp(theta2*x).*x]);
At=@(x,theta1,theta2) (transpose(J_x(x,theta1,theta2))*J_x...
    (x,theta1,theta2) );
gra=@(x,theta1,theta2,y) ( transpose(J_x(x,theta1,theta2))*f_x...
    (x,theta1,theta2,y) );
F_x=@(x,theta1,theta2,y) ( 0.5*transpose(f_x(x,theta1,theta2,y))*...
    f_x(x,theta1,theta2,y) );
%target function
Lmod=@(x,theta1,theta2,y,h) ( F_x(x,theta1,theta2,y)+h'*...
    transpose(J_x(x,theta1,theta2))*f_x(x,theta1,theta2,y)+...
    0.5*h'*At(x,theta1,theta2)*h );

Xdata = xdata';
Ydata = ydata';

v=2;
tau=1e-10;
k=0;
k_max=1000;
a0=theta0(1);b0=theta0(2);
t=[a0;b0];
Ndim=2;
A=At(Xdata,a0,b0);
g=gra(Xdata,a0,b0,Ydata);
found=(norm(g)<=epsilon);
mou=tau*max(diag(A));

%%%%%%%%%%%%%%%%%%%%%%%%%%%
while (~found &&(k<k_max))
    k=k+1;
    h_lm=-inv(A+mou*eye(Ndim))*g;
    if (norm(h_lm)<=epsilon*(norm(t)+epsilon))
        found=true;
    else
        t_new=t+h_lm;
        Fx=F_x(Xdata,t(1),t(2),Ydata);
        Fx_h=F_x(Xdata,t_new(1),t_new(2),Ydata) ;
        L_0=Lmod(Xdata,t(1),t(2),Ydata,zeros(Ndim,1));
        L_h=Lmod(Xdata,t(1),t(2), Ydata,h_lm);
        rho=(Fx-Fx_h)./(L_0-L_h);
        if rho>0
            t=t_new;
            A=At(Xdata,t(1),t(2));
            g=gra(Xdata,t(1),t(2),Ydata);
            found=(norm(g)<=epsilon);
            mou=mou*max([0.3333,1-(2*rho-1).^3]);
            v=2;
        else
            mou=mou*v;
            v=2*v;
        end
    end
vv=zeros(6);
for i=1:6
vv(i)=t(1)*exp(t(2)*Xdata(i))-Ydata(i);
end
val=0.5*(norm(vv))^2;
vallist(k+1)=val;
end
k;
vv0=zeros(1,k_max);
for i=1:6
vv0(i)=theta0(1)*exp(theta0(2)*Xdata(i))-Ydata(i);
end
vallist(1)=0.5*(norm(vv0))^2;
end
