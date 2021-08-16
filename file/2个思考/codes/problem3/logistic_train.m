function [omega,c,flist,glist,count] =...
    logistic_train(A,b,max_iter,epsilon,caseofmethod,caseoflinesearch)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch (caseofmethod)
case 1
    [omega,c,flist,glist,count] = logistic_Newton...
        (A,b,max_iter,epsilon,caseofmethod);
case 2
	[omega,c,flist,glist,count] = logistic_Steepest...
        (A,b,max_iter,epsilon,caseofmethod,caseoflinesearch);
case 3
    [omega,c,flist,glist,count] = logistic_CG...
        (A,b,max_iter,epsilon,caseofmethod,caseoflinesearch);
case 4
    [omega,c,flist,glist,count] = logistic_BFGS...
        (A,b,max_iter,epsilon,caseofmethod,caseoflinesearch);
case 5
    hatdelta=0.9;r0=0.1;eta=0.1;
    [omega,c,flist,glist,count] = logistic_Trustregion...
        (A,b,max_iter,epsilon,caseofmethod,r0,eta,hatdelta);

end

end


