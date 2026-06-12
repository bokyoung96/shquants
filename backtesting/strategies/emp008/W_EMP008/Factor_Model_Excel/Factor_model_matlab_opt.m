clc; clear;

% 파일 이름 한 곳에서 관리
filename = "Factor_Model_2025.07_수정.xlsm";

% 데이터 읽기
D          = xlsread(filename, "잔차", "D419:GU618");
z          = xlsread(filename, "temp", "CN3:DV202");
alpha_exp  = xlsread(filename, "Cov", "AO5:AO39");
cov        = xlsread(filename, "Cov", "C5:AK39");
index_wgt  = xlsread(filename, "backtest용_weight", "C22:C221");
n=26+1;
% 섹터제약 26, 롱숏 웨이트 1, 종목별투자제한 1
m=length(D);
TE=0.7/sqrt(12);

M=D+z*cov*z';

z_sec=zeros(26,m);
for i=1:26
    z_sec(i,:)=z(:,i+9);
end

fun=@(x)-alpha_exp'*z'*x;
Aeq=zeros(n,m);
Aeq(1,:)=ones(1,m);
% Aeq(2,61)=1; %종목투자제한용, 에코프로머티
% Aeq(3,75)=1; %종목투자제한용, 두산로보틱스
% Aeq(4,147)=1; %종목투자제한용, 금호타이어
% Aeq(6,75)=1;
% Aeq(7,80)=1;
% Aeq(8,92)=1;
% Aeq(9,157)=1;
% Aeq(10,160)=1;
% Aeq(11,167)=1;

for i=1:26
    Aeq(i+n-26,:)=z_sec(i,:);
end
    
beq=zeros(n,1);
% beq(2)=0;
% beq(3)=0;
% beq(2)=0;  종목투자제한용 뉴트럴
% beq(3)=0;  종목투자제한용
% beq(4)=0;  종목투자제한용
% beq(6)=-index_wgt(75);  편출 비우기용
% beq(7)=-index_wgt(80);
% beq(8)=-index_wgt(92);
% beq(9)=-index_wgt(157);
% beq(10)=-index_wgt(160);
% beq(11)=-index_wgt(167);

A=-eye(m,m);
x0=zeros(m,1);
options = optimoptions('fmincon','Display','iter','MaxFunctionEvaluations',1.0e8,'MaxIterations',1.0e8,'ConstraintTolerance', 1.0000e-10,'OptimalityTolerance',1.0000e-10,'StepTolerance',1.0000e-10);
x=fmincon(fun,x0,A,index_wgt,Aeq,beq,[],[],@(x)nonlcon(x,M,TE),options);

result=index_wgt+x;

function [c, ceq] = nonlcon(x, M, TE)
    % 비선형 부등식 제약조건: tracking error 제약
    % c ≤ 0 이어야 하므로, x' M x - TE^2 ≤ 0
    c = x' * M * x - TE^2;

    % 등식 제약 없음
    ceq = [];
end