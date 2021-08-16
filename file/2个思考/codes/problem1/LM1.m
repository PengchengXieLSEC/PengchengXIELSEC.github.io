function [x,val,k,vallist]=LM1(Fk,JFk,x0,epsilon,Xdata,Ydata)
maxk=100; 
rho=0.55;sigma=0.4; muk=norm(feval(Fk,x0));
k=0; n=length(x0);
vv0=zeros(1,maxk);
while(k<maxk)
    fk=feval(Fk,x0); 
    jfk=feval(JFk,x0); 
    gk=jfk'*fk;
    dk=-(jfk'*jfk+muk*eye(n))\gk;  
    if(norm(gk)<epsilon), 
        break; 
    end 
    m=0;mk=0;
    while(m<20) % Armijo
    newf=0.5*norm(feval(Fk,x0+rho^m*dk))^2; 
    oldf=0.5*norm(feval(Fk,x0))^2; 
	if(newf<oldf+sigma*rho^m*gk'*dk)
    	mk=m; 
		break;
    end
    m=m+1; 
end
x0=x0+rho^mk*dk; 
muk=norm(feval(Fk,x0));
val=0.5*muk^2;
vallist(k+2)=val;
for i=1:6
vv0(i)=0*exp(0*Xdata(i))-Ydata(i);
end
vallist(1)=0.5*(norm(vv0))^2;
k=k+1;
end
x=x0; 
