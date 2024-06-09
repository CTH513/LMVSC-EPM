clear;
clc;
warning off;
addpath(genpath('./'));

%% dataset
dataname = {'Caltech101-20'};
dsPath = './Dataset/';
resPath = './result/';
metric = {'ACC','nmi','Purity','Fscore','Precision','Recall','AR','Entropy'};

%% 
% load data & make folder
dataName = dataname{1};
disp(dataName);
load(strcat(dsPath,dataName));
%      X = fea;
%      Y = gt;

k = length(unique(Y));
matpath = strcat(resPath,dataName);
txtpath = strcat(resPath,strcat(dataName,'.txt'));

if (~exist(matpath,'file'))
    mkdir(matpath);
    addpath(genpath(matpath));
end

dlmwrite(txtpath, strcat('Dataset:',cellstr(dataName), '  Date:',datestr(now)),'-append','delimiter','','newline','pc');

%% parameters setting
m = [k,2*k,3*k];
d = [k,2*k,3*k,4*k,5*k];
lambda = [0.0001,0.0001,0.001,0.01,0.1,1,10];
beta = [0.0001,0.0001,0.001,0.01,0.1,1,10];

%%
for ichor = 1:length(m)
    for id = 1:length(d)
        for j = 1:length(lambda)
            for i = 1:length(beta)
                tic;
                [A,P,Z,G,F,iter,obj,alpha] = algo_LMVSC_EPM(X,Y,lambda(j),beta(i),d(id),m(ichor)); % X,Y,lambda,beta,d,m
                [~,idx]=max(F);
                res = Clustering8Measure(Y,idx); % [ACC nmi Purity Fscore Precision Recall AR Entropy]
                timer(ichor,id)  = toc;
                str = strcat('m:',num2str(m(ichor)),'       lambda:',num2str(lambda(j)),'         beta:',num2str(beta(i)),'      d:',num2str(d(id)),'     res:',num2str(res),'       Time:',num2str(timer(ichor,id)));
                disp(str);
                dlmwrite(txtpath, [lambda(j) beta(i) m(ichor) d(id) res timer(ichor,id)],'-append','delimiter','\t','newline','pc');
                matname = ['_Anchor_',num2str(m(ichor)),'_Dimensionality_',num2str(d(id)),'.mat'];
                save([matpath,'/',matname],'P','A','Z','alpha');
            end
        end
    end
end
clear resall objall X Y k



