%gradient
function gradent_omegaandc=gradientfun(n,m,omegaandc,a,b)  
gradent_omegaandc = zeros(n+1,1);
    for i=1:m
        aa = exp(-b(i)*(omegaandc'*a(:,i)));
        gradent_omegaandc = gradent_omegaandc +...
            1/m*(aa)/(1+aa)*(-b(i).*a(:,i));
    end
end
