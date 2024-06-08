function [A,P,Z,G,F,iter,obj,alpha] = algo_LMVSC_EPM(X,Y,lambda,beta,d,m)
% d      : the number of dimensionality. the size of P is di*d  
% m      : the number of anchor. the size of Z is m*n.
% lambda : the hyper-parameter of regularization term.
% beta   : the hyper-parameter of regularization term.
% X      : n*di

%% initialize
maxIter = 30 ; % the number of iterations

% mu = 1e-4;
numclass = length(unique(Y));
numview = length(X);
numsample = size(Y,1);

P = cell(numview,1);    % di * d
A = zeros(d,m);         % d  * m
Z = zeros(m,numsample); % m  * n

for i = 1:numview
   di = size(X{i},1); 
   P{i} = zeros(di,d);
   X{i} = mapstd(X{i}',0,1); 
end
Z(:,1:m) = eye(m);
G = eye(m,numclass);
F = eye(numclass,numsample); 

alpha = ones(1,numview)/numview;
opt.disp = 0;

flag = 1;
iter = 0;
%%
while flag
    iter = iter + 1;
    
    %% optimize P_i
    parfor iv=1:numview
        C = X{iv}*Z'* A';
        [U,~,V] = svd(C,'econ');
        P{iv} = U*V';
    end

    %% optimize A
    sumAlpha = 0;
    part1 = 0;
    for ia = 1:numview
        al2 = alpha(ia)^2;
        sumAlpha = sumAlpha + al2;
        part1 = part1 + al2 * P{ia}' * X{ia} * Z';
    end
    [U1,~,V1] = svd(part1,'econ');
    A = U1*V1';
    
    %% optimize Z
    H = 2*sumAlpha*eye(m)+2*lambda*eye(m)+2*beta*eye(m);
    H = (H+H')/2;
    options = optimset( 'Algorithm','interior-point-convex','Display','off'); 
    parfor ji=1:numsample
        ff=0;
        e = F(:,ji)'*G';
        for j=1:numview
            C = P{j} * A;
            ff = ff - 2*X{j}(:,ji)'*C - 2*lambda*e;
        end
        Z(:,ji) = quadprog(H,ff',[],[],ones(1,m),1,zeros(m,1),ones(m,1),[],options);
    end

    %% optimize G
    J = Z*F';      
    [U2,~,V2] = svd(J,'econ');
    G = U2*V2';

    %% optimize F
    F=zeros(numclass,numsample);
    for i=1:numsample
        Dis=zeros(numclass,1);
        for j=1:numclass
            Dis(j)=(norm(Z(:,i)-G(:,j)))^2;
        end
        [~,r]=min(Dis);
        F(r(1),i)=1;
    end

    %% optimize alpha
    M = zeros(numview,1);
    for iv = 1:numview
        M(iv) = norm( X{iv} - P{iv} * A * Z,'fro')^2;
    end
    Mfra = M.^-1;
    Q = 1/sum(Mfra);
    alpha = Q*Mfra;

    %%
    term1 = 0;
    for iv = 1:numview
        term1 = term1 + alpha(iv)^2 * norm(P{iv}' * X{iv} - A * Z,'fro')^2;
    end
    term2 = lambda * norm(Z - G * F,'fro')^2;
    term3 = beta * norm(Z,'fro')^2;
    obj(iter) = term1+ term2+term3;
    
    if (iter>=20) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-6 || iter>maxIter || obj(iter) < 1e-15) 
        flag = 0;
    end
end
         
         
    
