%Hessian
function hessian_omegaandc=hessianfun(n,m,omegaandc,a,b)
hessian_omegaandc = zeros(n+1,n+1);
 for i=1:m
        aa = exp(-b(i)*(omegaandc'*a(:,i)));
        hessian_omegaandc = hessian_omegaandc +...
            1/m*(aa)/(1+aa)^2*(-b(i))^2.*(a(:,i)*a(:,i)');
    end
end