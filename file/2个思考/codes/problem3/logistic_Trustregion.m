function [omega,c,flist,glist,count] =...
    logistic_Newton(A,b,max_iter,epsilon,caseofmethod,r0,eta,hatdelta)

% Trust Region Method
[n m]= size(A);
A=[A;ones(1,m)];
omegaandc = ones(n+1,1)/(n+1);
count = 1;
flist=zeros(1,max_iter);
gradient_omegaandc = zeros(n+1,1);
hessian_omegaandc = zeros(n+1,n+1);
I=eye(n+1);
r=r0;
flist(1)=targetfunction(n,m,omegaandc,A,b);
glist=zeros(1,max_iter);
glist(1)=norm(gradientfun(n,m,omegaandc,A,b));
%main
while count < max_iter
    %get gradient
    gradient_omegaandc = gradientfun(n,m,omegaandc,A,b);

    %get Hessian
    hessian_omegaandc = hessianfun(n,m,omegaandc,A,b);
    
    if norm(gradient_omegaandc,2) <= epsilon
            fprintf('finished seraching the step size\n');
            break;
    end
   
    w=hessian_omegaandc*gradient_omegaandc;
    p_U=(-gradient_omegaandc'*gradient_omegaandc)/...
        (gradient_omegaandc'*w).*gradient_omegaandc;
    p_B=-(hessian_omegaandc+1e-10*I)\gradient_omegaandc;
    
    if norm(p_U)>=r
    p=(r/norm(p_U)).*p_U;
    elseif norm(p_B)<=r
        p=p_B;
    else 
        p_s=p_B-p_U;
        a=norm(p_s)^2;
        bfortr=2*p_U'*p_s;
        c=norm(p_U)^2-r^2;
        alpha=roots([a bfortr c]);
        alpha=alpha(find(alpha>0));
        p=p_U+alpha.*p_s;
end
   
    %get next iterate point
    omegaandcnew = omegaandc + p;
    
    v=hessian_omegaandc*gradient_omegaandc;
    rho1=targetfunction(n,m,omegaandc,A,b)-...
        targetfunction(n,m,omegaandcnew,A,b);
    rho2=-0.5*p'*v-gradient_omegaandc'*p;
    rho=rho1/rho2;
    if rho>0.75&&norm(p)==r
         r=min([2*r,hatdelta]);
    elseif rho<0.25
         r=0.25*r;
    end
    if rho<eta
        omegaandcnew = omegaandc;
    end 
    omegaandc = omegaandcnew;
    
    %list of function value
    flist(count+1)=targetfunction(n,m,omegaandc,A,b);
    glist(count+1)=norm(gradientfun(n,m,omegaandc,A,b));
    count = count + 1;
    
    %get no good solution
      if  count > max_iter-1
       fprintf('not converged to desired precision!');
        break;
      end
    
end
omega=omegaandc(1:n);
c=omegaandc(n+1);
end
