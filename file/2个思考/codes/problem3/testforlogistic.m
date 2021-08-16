%TEST FOR LOGISTIC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                              Method Choice
%caseofmethod 1.Newton 2.SD 3.CG 4.BFGS 5.Trust Region Dogleg
%
%caseoflinesearch 1.Inexact line search 2.fminsearch Exact line search
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc

Timeornot=0;%time
Plot1ornot=1;%1
Plot2ornot=1;%2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m=500;
n=1000;
max_iterate=10000;
caseofmethod=5;
caseoflinesearch=1;
epsilon=1.0e-4;
CPUtime=zeros(10);
A=randn(n,m);
b=sign(rand(m,1)-0.5);

CPUaver=zeros(5,2);
COUNTaver=zeros(5,2);

%{
%testnow

[omega,c,flist,gflist,count] =...
logistic_train(A,b,max_iterate,epsilon,5,2);
%}    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if Timeornot==1
N=3;
for caseofmethod=1:5
    for caseoflinesearch=1:2
         if ((caseofmethod==1)&&(caseoflinesearch==2))...
                 ||((caseofmethod==5)&&(caseoflinesearch==2))
             break
         else
            for k=1:N
            A=randn(n,m);
            b=sign(rand(m,1)-0.5);
            t1=clock;
            [omega,c,flist,count] = logistic_train...
                (A,b,max_iterate,epsilon,caseofmethod,caseoflinesearch);
            t2=clock;
            CPUtime(k)=etime(t2,t1);
            COUNT(k)=count;
            end
            CPUaver(caseofmethod,caseoflinesearch)=1/N*sum(CPUtime(:));
            COUNTaver(caseofmethod,caseoflinesearch)=1/N*sum(COUNT(:));
         end
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
if Plot1ornot==1
[omega_1,c_1,flist_1,glist_1,count_1] =...
    logistic_train(A,b,max_iterate,epsilon,1);
[omega_2_1,c_2_1,flist_2_1,glist_2_1,count_2_1] = logistic_train...
    (A,b,max_iterate,epsilon,2,1);
[omega_2_2,c_2_2,flist_2_2,glist_2_2,count_2_2] = logistic_train...
    (A,b,max_iterate,epsilon,2,2);
[omega_3_1,c_3_1,flist_3_1,glist_3_1,count_3_1] = logistic_train...
    (A,b,max_iterate,epsilon,3,1);
[omega_3_2,c_3_2,flist_3_2,glist_3_2,count_3_2] = logistic_train...
    (A,b,max_iterate,epsilon,3,2);
[omega_4_1,c_4_1,flist_4_1,glist_4_1,count_4_1] = logistic_train...
    (A,b,max_iterate,epsilon,4,1);
[omega_4_2,c_4_2,flist_4_2,glist_4_2,count_4_2] = logistic_train...
    (A,b,max_iterate,epsilon,4,2);
[omega_5,c_5,flist_5,glist_5,count_5] = logistic_train...
    (A,b,max_iterate,epsilon,5);
%}
hold on
plot(0:count_1+1,flist_1(1,1:count_1+2),'O-','linewidth',3);
plot(0:count_2_1+1,flist_2_1(1,1:count_2_1+2),'O-','linewidth',3);
plot(0:count_2_2+1,flist_2_2(1,1:count_2_2+2),'O-','linewidth',3);
plot(0:count_3_1+1,flist_3_1(1,1:count_3_1+2),'O-','linewidth',3);
plot(0:count_3_2+1,flist_3_2(1,1:count_3_2+2),'O-','linewidth',3);
plot(0:count_4_1+1,flist_4_1(1,1:count_4_1+2),'O-','linewidth',3);
plot(0:count_4_2+1,flist_4_2(1,1:count_4_2+2),'O-','linewidth',3);
plot(0:count_5+1,flist_5(1,1:count_5+2),'O-','linewidth',3);
legend('Newton','SD (Inexact LS)',...
    'SD (Exact LS)',...
    'CG (Inexact LS)',...
    'CG (Exact LS)',...
    'BFGS (Inexact LS)','BFGS (Exact LS)',...
    'TR (Dogleg)','FontSize',30);
xlabel('Iteration Number','FontSize',30);
ylabel('$f(w,c)$','Interpreter','latex','FontSize',30);
title('Logistic regression problem solved by different algorithms',...
    'FontSize',30);
set(gca,'linewidth',2,'fontsize',30,'fontname','Times');
set(gcf,'position',[0.5,0.5,2100,1200]);
print -depsc -r300 plot/epsilon=10^-4/Log_4_6
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
if Plot2ornot==1
    %{
[omega_1,c_1,flist_1,glist_1,count_1] = ...
logistic_train(A,b,max_iterate,epsilon,1);
[omega_2_1,c_2_1,flist_2_1,glist_2_1,count_2_1] = logistic_train...
    (A,b,max_iterate,epsilon,2,1);
[omega_2_2,c_2_2,flist_2_2,glist_2_2,count_2_2] = logistic_train...
    (A,b,max_iterate,epsilon,2,2);
[omega_3_1,c_3_1,flist_3_1,glist_3_1,count_3_1] = logistic_train...
    (A,b,max_iterate,epsilon,3,1);
[omega_3_2,c_3_2,flist_3_2,glist_3_2,count_3_2] = logistic_train...
    (A,b,max_iterate,epsilon,3,2);
[omega_4_1,c_4_1,flist_4_1,glist_4_1,count_4_1] = logistic_train...
    (A,b,max_iterate,epsilon,4,1);
[omega_4_2,c_4_2,flist_4_2,glist_4_2,count_4_2] = logistic_train...
    (A,b,max_iterate,epsilon,4,2);
[omega_5,c_5,flist_5,glist_5,count_5] = logistic_train...
    (A,b,max_iterate,epsilon,5);
    %}
hold on
plot(0:count_1+1,glist_1(1,1:count_1+2),'O-','linewidth',3);
plot(0:count_2_1+1,glist_2_1(1,1:count_2_1+2),'O-','linewidth',3);
plot(0:count_2_2+1,glist_2_2(1,1:count_2_2+2),'O-','linewidth',3);
plot(0:count_3_1+1,glist_3_1(1,1:count_3_1+2),'O-','linewidth',3);
plot(0:count_3_2+1,glist_3_2(1,1:count_3_2+2),'O-','linewidth',3);
plot(0:count_4_1+1,glist_4_1(1,1:count_4_1+2),'O-','linewidth',3);
plot(0:count_4_2+1,glist_4_2(1,1:count_4_2+2),'O-','linewidth',3);
plot(0:count_5+1,glist_5(1,1:count_5+2),'O-','linewidth',3);
legend('Newton','SD (Inexact LS)',...
    'SD (Exact LS)',...
    'CG (Inexact LS)',...
    'CG (Exact LS)',...
    'BFGS (Inexact LS)','BFGS (Exact LS)',...
    'TR (Dogleg)','FontSize',30);
xlabel('Iteration Number','FontSize',30);
ylabel('$\Vert \nabla f(w,c)\Vert$','Interpreter','latex','FontSize',30);
title('Logistic regression problem solved by different algorithms',...
    'FontSize',30);
set(gca,'linewidth',2,'fontsize',30,'fontname','Times');
set(gcf,'position',[0.5,0.5,2100,1200]);
print -depsc -r300 plot/epsilon=10^-4/GLog_4_6
end
