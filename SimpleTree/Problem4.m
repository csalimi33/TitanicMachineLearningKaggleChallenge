X = csvread('fairAge.csv',1,0); % start reading from second row and first column
Survived = X(:,1); % faster than a separate csvread
X(:,1) = [];
Age=X(:,1);
Fare=X(:,2);

%FareRange=range(Fare);
%meanFare=mean(Fare);
%Fare=((Fare- 32.2525)/ 0.0971);

test=csvread('FairAgeTest.csv',1,0);
Age1 = test(:,1);
Fare1 = test(:,2);
%Fare1=Fare1/FareRange;
%Fare1=((Fare1- 32.2525)/ 0.0971);

rng(1); % For reproducibility
MdlDefault = fitctree([Age,Fare],Survived,'CrossVal','on');

numBranches = @(x)sum(x.IsBranch);
mdlDefaultNumSplits = cellfun(numBranches, MdlDefault.Trained);



%Regularized and Normalized all give the same CrossVal accuracy

Mdl7 = fitctree([Age, Fare],Survived, 'CrossVal', 'on', ...
                 'MaxNumSplits',22,'MinLeafSize',10,...
                 'PredictorNames',{'Age','Fair'},'CategoricalPredictors',...
                 'Age');


classErrorDefault = kfoldLoss(MdlDefault);
classError7 = kfoldLoss(Mdl7);
classError7


%fit a separate model here, varying the above parameters so that crossval
%is off, you can't predict unless crossval is off
MdlPredict = fitctree([Age, Fare],Survived,...
                     'MaxNumSplits',22,'MinLeafSize',10,...
                 'PredictorNames',{'Age','Fair'},'CategoricalPredictors',...
                 'Age');


simpleTree = predict(MdlPredict,[Age1, Fare1]);

%Messing with bagging code stuff...
%B = TreeBagger(7,[Age,Fare,Embarked,Parch,Pclass,Sex,SibSp]...
 %   ,Survived, 'OOBPred', 'on');







