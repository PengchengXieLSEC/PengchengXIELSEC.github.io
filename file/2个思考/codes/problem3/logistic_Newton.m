function [omega,c,flist,glist,count] =...
    logistic_Newton(A,b,max_iter,epsilon,caseofmethod,caseoflinesearch)

% Newton's Method
[n m]= size(A);
A=[A;ones(1,m)];
omegaandc = ones(n+1,1)/(n+1);
count = 1;
I=eye(n+1);
flist=zeros(1,max_iter);
glist=zeros(1,max_iter);
gradient_omegaandc = zeros(n+1,1);
hessian_omegaandc = zeros(n+1,n+1);
flist(1)=targetfunction(n,m,omegaandc,A,b);
glist(1)=norm(gradientfun(n,m,omegaandc,A,b));

while count < max_iter
    %get gradient
    gradient_omegaandc = gradientfun(n,m,omegaandc,A,b);

    %get Hessian
    hessian_omegaandc = hessianfun(n,m,omegaandc,A,b);
    
    if norm(gradient_omegaandc,2) <= epsilon
            fprintf('finished seraching the step size\n');
            break;
    end
   
    %get direction
    direction=-gradient_omegaandc;
    stepsize=(hessian_omegaandc+1e-10*I)^(-1);
    
    %get next iterate point
    omegaandcnew = omegaandc + stepsize*direction;
    omegaandc = omegaandcnew;
    
    %list of function value
    glist(count+1)=norm(gradientfun(n,m,omegaandc,A,b));
    flist(count+1)=targetfunction(n,m,omegaandc,A,b);
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


