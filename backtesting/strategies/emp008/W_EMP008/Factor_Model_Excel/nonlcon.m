function[c,ceq]=nonlcon(x,M,d)
c=[];
ceq=sqrt(x'*M*x)-d;