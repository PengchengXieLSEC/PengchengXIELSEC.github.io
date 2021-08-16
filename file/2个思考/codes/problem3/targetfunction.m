%Function value
function f=targetfunction(n,m,omegaandc,a,b)
f=0;
   for i=1:m
       aa = exp(-b(i)*(omegaandc'*a(:,i)));
        f = f+1/m*log(1+aa);       
   end
end
