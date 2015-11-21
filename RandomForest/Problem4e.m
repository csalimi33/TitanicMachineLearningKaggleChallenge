%--------------------------------------------------------------------------
%                                                                         %
%      Problem 4.e Random Forest Prediction                               %
%                                                                         %
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


rng(1); % For reproducibility
MdlDefault = fitctree([Xvar, Fare],Survived,'CrossVal','on',...
                 'PredictorNames',{'Xvar','Fair','W','C','W','W','W','W',...
                 'W','W','W','t','e'},'CategoricalPredictors',...
                 'Xvar');

             
numBranches = @(x)sum(x.IsBranch);
mdlDefaultNumSplits = cellfun(numBranches, MdlDefault.Trained);


MdlPredict = TreeBagger(29,[Xvar, Fare],Survived,'OOBPred','on','Method',...
                  'classification','NVarToSample','all',...
                  'MaxNumSplits',24,'MinLeafSize',12,...
                 'PredictorNames',{'Xvar','Fair','W','C','W','W','W','W',...
                 'W','W','W','t','e'},'CategoricalPredictors',...
                 'Xvar');


oobErrorBaggedEnsemble = oobError(MdlPredict);
oobErrorBaggedEnsemble
MdlRandomForest = TreeBagger(29,[Xvar, Fare],Survived,'Method',...
                  'classification','NVarToSample','all',...
                  'MaxNumSplits',24,'MinLeafSize',12,...
                 'PredictorNames',{'Xvar','Fair','W','C','W','W','W','W',...
                 'W','W','W','t','e'},'CategoricalPredictors',...
                 'Xvar');

             
XvarTest = csvread('PreProcessedTest2.csv',1,0); % start reading from second row and first column
FareTest=XvarTest(:,6);
XvarTest(:,6)=[];


%--------------------------------------------------------------------------
%                                                                         %
%      Make Prediction                                                    %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %                                                                         
%                                                                         %
%                                                                         %
%--------------------------------------------------------------------------
RandomForestPredictions = predict(MdlRandomForest,[XvarTest, FareTest]);




