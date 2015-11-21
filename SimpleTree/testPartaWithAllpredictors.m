%--------------------------------------------------------------------------
%                                                                         %
%      Test part a with all predictors                                                        %
%                                                                         %
%                                                                         %
%--------------------------------------------------------------------------
X = csvread('PreProcessedTrain2.csv',1,0); % start reading from second row and first column
Survived = X(:,1); % faster than a separate csvread
X(:,1) = [];
Xvar = csvread('PreProcessedTrain2.csv',1,0); % start reading from second row and first column
Xvar(:,1)=[];
Xvar(:,6)=[];

Fare=X(:,6);

XvarTest = csvread('PreProcessedTest2.csv',1,0); % start reading from second row and first column
FareTest=XvarTest(:,6);
XvarTest(:,6)=[];


MdlDefault = fitctree([Xvar,Fare],Survived,'CrossVal','on');

numBranches = @(x)sum(x.IsBranch);
mdlDefaultNumSplits = cellfun(numBranches, MdlDefault.Trained);


%Regularized and Normalized all give the same CrossVal accuracy


Mdl7 = fitctree([Xvar, Fare],Survived,'MaxNumSplits',22,'MinLeafSize',10,...
                 'CrossVal','on',...
                 'PredictorNames',{'Xvar','Fair','W','C','W','W','W','W',...
                 'W','W','W','t','e'},'CategoricalPredictors',...
                 'Xvar');



classErrorDefault = kfoldLoss(MdlDefault);
classError7 = kfoldLoss(Mdl7);
classError7


%fit a separate model here, varying the above parameters so that crossval
%is off, you can't predict unless crossval is off
MdlPredict = fitctree([Xvar, Fare],Survived,'MaxNumSplits',19,'MinLeafSize',12,...
                 'PredictorNames',{'Xvar','Fair','W','C','W','W','W','W',...
                 'W','W','W','t','e'},'CategoricalPredictors',...
                 'Xvar');;

view(MdlPredict, 'mode', 'graph');

%--------------------------------------------------------------------------
%                                                                         %
%      Problem 4.b, Kaggle Accuracy =78%                                                         %
%                                                                         %
%                                                                         %
%                                                                         %
%--------------------------------------------------------------------------
simpleTree = predict(MdlPredict,[XvarTest, FareTest]);

%Messing with bagging code stuff...
%B = TreeBagger(7,[Age,Fare,Embarked,Parch,Pclass,Sex,SibSp]...
 %   ,Survived, 'OOBPred', 'on');







