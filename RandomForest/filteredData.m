

%ntrees=40;

%treeBag = TreeBagger(ntrees,X ,Y);

%predictBag = predict(treeBag,Xtest);


%r=rand(885,1)<0.90;
%Xtrain = X(r,:);
%Ytrain = Y(r,:);

%Xvalid=X(~r,:);
%Yvalid=Y(~r,:);

%cv =cvpartition(X, 'holdout',0.1);

%
%t = ClassificationTree.template('MinLeaf',5);
%mdl = fitensemble(Xtrain,Ytrain,'LPBoost',500,t,...
  %  'PredictorNames',{'W','C','W','W','W','W','W','W','W',...
   %              'W','W','W','W','W','W','W','W','W','W','W','W',...
    %             'W','W','W','W','W','W',},'ResponseName','Survived');

%L = loss(mdl,Xvalid, Yvalid ,'mode','ensemble');
%fprintf('Mean-square testing error = %f\n',L);

%mdl1 = regularize(mdl,'lambda',[0.001 0.1]);
%disp('Number of Trees:');
%disp(sum(mdl1.Regularization.TrainedWeights > 0));

%mdl = shrink(mdl1,'weightcolumn',2);

%disp('Number of Trees trained after shrinkage');
%disp(mdl1.NTrained);

%predictBoost = predict(mdl1,Xtest);


%mdlRandomForest = fitensemble(Xtrain,Ytrain,'NVarToSample', 'all','LPBoost',500,t,...
 %   'PredictorNames',{'W','C','W','W','W','W','W','W','W',...
  %               'W','W','W','W','W','W','W','W','W','W','W','W',...
   %              'W','W','W','W','W','W',},'ResponseName','Survived');


