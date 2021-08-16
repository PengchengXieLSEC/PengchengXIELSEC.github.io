function [omega,c,flist,glist,count] ...
    = logistic_train(A,b,max_iter,epsilon,caseofmethod,caseoflinesearch)

%CG

[n m] = size(A);
A=[A;ones(1,m)];
omegaandc = ones(n+1,1)/(n+1);
count = 1;
I=eye(n+1);
flist=zeros(1,max_iter);
alphamin=0;alphamax=NaN;
rho=2; alpha=50; sigma=0.5e-4;sigma2=1.0e-4;
glist=zeros(1,max_iter);
glist(1)=norm(gradientfun(n,m,omegaandc,A,b));
nforCG=length(omegaandc);
partial_beta_old=1;
direction_old=1; 
gradient_omegaandc = zeros(n+1,1);
flist(1)=targetfunction(n,m,omegaandc,A,b);
if caseoflinesearch==1
while count < max_iter
    
     %get gradient
     gradient_omegaandc = gradientfun(n,m,omegaandc,A,b);
     
     if norm(gradient_omegaandc,2)<=epsilon
            fprintf('finished seraching the step size\n');
            break;
     end
     
     %get direction
     iterforCG=count-(nforCG+1)*floor(count/(nforCG+1));
     iterforCG=iterforCG+1;
     if(iterforCG==1)
        direction=-gradient_omegaandc; 
        partial_beta_old=gradient_omegaandc;
        direction_old=-gradient_omegaandc;
     else
        betaforCG=(gradient_omegaandc'*gradient_omegaandc)/...
            (partial_beta_old'*partial_beta_old); 
        direction=-gradient_omegaandc+ betaforCG*direction_old; 
        gd=gradient_omegaandc'*direction; 
        if(gd'>=0.0)
        direction=-gradient_omegaandc; 
        end
     end
     
       %get step size:  Wolfe
      while 1
      funold=targetfunction(n,m,omegaandc,A,b);
      funnew=targetfunction(n,m,omegaandc+alpha*direction,A,b);
      gradold=gradientfun(n,m,omegaandc,A,b);
      gradnew=gradientfun(n,m,omegaandc+alpha*direction,A,b);
        if funnew<funold+sigma*alpha*gradold'*direction
            if gradnew'*direction>sigma2*alpha*gradold'*direction
                break
            else
                alphamin=alpha;
                if alphamax<NaN
                 alpha=(alphamin+alphamax)/2;
                else 
                 alpha=rho*alpha;
                end
            end
        else
        alphamax=alpha;
        alpha=(alphamax+alphamin)/2;
        end
      end
       stepsize=alpha;
       
        %get next iterate point
        omegaandcnew = omegaandc + stepsize*direction;
        omegaandc = omegaandcnew;
        
    flist(count+1)=targetfunction(n,m,omegaandc,A,b);
    glist(count+1)=norm(gradientfun(n,m,omegaandc,A,b));
    
    partial_beta_old=gradient_omegaandc; 
    direction_old=direction;
    count = count + 1;
    
    %get no good solution
    if  count > max_iter
       fprintf('not converged to desired precision!');
        break;
    end

end
else
    while count < max_iter
    
     %get gradient
     gradient_omegaandc = gradientfun(n,m,omegaandc,A,b);
     
     if norm(gradient_omegaandc,2)<= epsilon
            fprintf('finished seraching the step size\n');
            break;
     end
     
     %get direction
     iterforCG=count-(nforCG+1)*floor(count/(nforCG+1));
     iterforCG=iterforCG+1;
     if(iterforCG==1)
        direction=-gradient_omegaandc; 
        partial_beta_old=gradient_omegaandc;
        direction_old=-gradient_omegaandc;
     else
        betaforCG=(gradient_omegaandc'*gradient_omegaandc)/...
            (partial_beta_old'*partial_beta_old); 
        direction=-gradient_omegaandc+ betaforCG*direction_old; 
        gd=gradient_omegaandc'*direction; 
        if(gd'>=0.0)
        direction=-gradient_omegaandc; 
        end
     end
     
        %get step size: fminsearch
        stepsize=fminsearch(@(stepsize)(targetfunction(n,m,omegaandc+...
            stepsize*direction,A,b)),0); 
     
        %get next iterate point
        omegaandcnew = omegaandc + stepsize*direction;
        omegaandc = omegaandcnew;
        
    flist(count+1)=targetfunction(n,m,omegaandc,A,b);
    glist(count+1)=norm(gradientfun(n,m,omegaandc,A,b));
    partial_beta_old=gradient_omegaandc; 
    direction_old=direction;
    count = count + 1;
    
    %get no good solution
    if  count > max_iter
       fprintf('not converged to desired precision!');
        break;
    end

end
end
omega=omegaandc(1:n);
c=omegaandc(n+1);

end



