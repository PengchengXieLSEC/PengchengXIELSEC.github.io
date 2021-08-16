function [omega,c,flist,glist,count] =...
    logistic_train(A,b,max_iter,epsilon,caseofmethod,caseoflinesearch)
%BFGS
[n m] = size(A);
A=[A;ones(1,m)];
omegaandc = ones(n+1,1)/(n+1);
count = 1;
I=eye(n+1);
flist=zeros(1,max_iter);
rho=0.9;sigma=0.48;%Armijo
glist=zeros(1,max_iter);
glist(1)=norm(gradientfun(n,m,omegaandc,A,b));
nforCG=length(omegaandc);
partial_beta_old=1;
direction_old=1; 
gradient_omegaandc = zeros(n+1,1);
Bk=eye(n+1);
flist(1)=targetfunction(n,m,omegaandc,A,b);
if caseoflinesearch==1
while count < max_iter
    
     %get gradient
     gradient_omegaandc = gradientfun(n,m,omegaandc,A,b);
     
     if norm(gradient_omegaandc,2)<= epsilon
            fprintf('finished seraching the step size\n');
            break;
     end
     
     %get direction
     direction=-Bk\gradient_omegaandc; 
     
     %get step size:

     % Armijo
     mm=0; mk=0;
     while(mm<20) 
         funold=targetfunction(n,m,omegaandc,A,b);
         funnew=targetfunction(n,m,omegaandc+rho^mm*direction,A,b);
         if(...
            funnew<funold+sigma*rho^mm*gradient_omegaandc'*direction...
           )
         mk=mm; 
         break;
         end
         mm=mm+1; 
     end   
        stepsize=rho^mk; 

  

        %get next iterate point
        omegaandcnew = omegaandc +stepsize*direction;
        %BFGS
        gradient_omegaandcnew = gradientfun(n,m,omegaandcnew,A,b);
        sk=omegaandcnew-omegaandc; 
        yk=gradient_omegaandcnew-gradient_omegaandc;
        if(yk'*sk>0)
        Bk=Bk-(Bk*sk*sk'*Bk)/(sk'*Bk*sk)+(yk*yk')/(yk'*sk); 
        end
        
        omegaandc = omegaandcnew;
        
    flist(count+1)=targetfunction(n,m,omegaandc,A,b);
    glist(count+1)=norm(gradientfun(n,m,omegaandc,A,b));

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
     direction=-Bk\gradient_omegaandc; 
     
        %get step size: fminsearch
        stepsize=fminsearch(@(stepsize)(targetfunction...
            (n,m,omegaandc+stepsize*direction,A,b)),0); 
     
        %get next iterate point
        omegaandcnew = omegaandc +stepsize*direction;
        %BFGS
        gradient_omegaandcnew = gradientfun(n,m,omegaandcnew,A,b);
        sk=omegaandcnew-omegaandc; 
        yk=gradient_omegaandcnew-gradient_omegaandc;
        if(yk'*sk>0)
        Bk=Bk-(Bk*sk*sk'*Bk)/(sk'*Bk*sk)+(yk*yk')/(yk'*sk); 
        end
        
        omegaandc = omegaandcnew;
        
        
    flist(count+1)=targetfunction(n,m,omegaandc,A,b);
    glist(count+1)=norm(gradientfun(n,m,omegaandc,A,b));
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




