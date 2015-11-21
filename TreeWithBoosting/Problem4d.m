
%--------------------------------------------------------------------------
%                                                                         %
%      Problem 4.d, Boosting, Kaggle Accuracy =%                                                         %
%                                                                         %
%                                                                         %
%                                                                         %
%--------------------------------------------------------------------------

X = csvread('PreProcessedTrain2.csv',1,0); % start reading from second row and first column
n=length(X);
% Set aside 90% of the data for training
cv = cvpartition(n,'holdout',0.1);
Xvar=X(cv.training,:);
Survived = Xvar(:,1); % faster than a separate csvread
Xvar(:,1) = [];
Fare=Xvar(:,6);
Xvar(:,6) = [];



XvarVal=X(cv.test,:);
SurvivedTest=XvarVal(:,1);
XvarVal(:,1) = [];
FareTest=XvarVal(:,6);
XvarVal(:,6) = [];






t = ClassificationTree.template('MinLeaf',4);

rng(1); % For reproducibility
MdlBoost = fitensemble([Xvar, Fare],Survived,'RUSBoost',5,t,...
                 'PredictorNames',{'Xvar','Fair','W','C','W','W','W','W',...
                 'W','W','W','t','e'},'CategoricalPredictors',...
                 'Xvar','ResponseName', 'Survived','Method',...
                 'classification');
 
L = loss(MdlBoost,[XvarVal,FareTest],SurvivedTest,'mode','ensemble');
fprintf('Mean-square testing error = %f\n',L);


%--------------------------------------------------------------------------
%                                                                         %
%      Make Prediction                                                    %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %                                                                         
%                                                                         %
%                                                                         %
%--------------------------------------------------------------------------
% XvarTest = csvread('PreProcessedTest2.csv',1,0); % start reading from second row and first column
% FareTest=XvarTest(:,6);
% XvarTest(:,6)=[];
% 
% 
% ens = predict(MdlBoost,[XvarTest, FareTest]);
