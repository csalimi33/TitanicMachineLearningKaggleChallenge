%--------------------------------------------------------------------------
%                                                                         %
%      Problem 4.e Random Forest Prediction                               %
%                                                                         %
%                                                                         %
%                                                                         %
%--------------------------------------------------------------------------


Fare=Xvar(7,:);

Xvar(7,:)=[];

rng(1); % For reproducibility
MdlDefault = fitctree([Xvar, Fare],Y,'CrossVal','on',...
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


mdlRandomForest = TreeBagger(29,[Xvar,Fare],Y,'NVarToSample', 'all',...
            'LPBoost',500,t,'Method', 'classification','MaxNumSplits',24,...
            'MinLeafSize',12,...
            'PredictorNames',{'W','C','W','W','W','W','W','W','W',...
                 'W','W','W','W','W','W','W','W','W','W','W','W',...
                 'W','W','W','W','W','W',},'ResponseName','Survived');
   
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




