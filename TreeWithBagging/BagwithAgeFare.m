%--------------------------------------------------------------------------
%                                                                         %
%      Problem 4.a                                                        %
%                                                                         %
%                                                                         %
%                                                                         %
%--------------------------------------------------------------------------
X = csvread('PreProcessedTrain2.csv',1,0); % start reading from second row and first column

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




%Regularized and Normalized all give the same CrossVal accuracy



AgeTreeBag = TreeBagger(500,[Age, Fare],Survived,'OOBPred','on','Method',...
                 'classification','MaxNumSplits',19,'MinLeafSize',12,...
                 'PredictorNames',{'Age','Fair'},'CategoricalPredictors',...
                 'Age');

oobErrorBaggedEnsemble = oobError(AgeTreeBag);
oobErrorBaggedEnsemble




%fit a separate model here, varying the above parameters so that crossval
%is off, you can't predict unless crossval is off

MdlRandomForest = TreeBagger(43,[Xvar, Fare],Survived,'Method',...
                     'classification','MaxNumSplits',19,'MinLeafSize',12,...
                 'PredictorNames',{'Xvar','Fair','W','C','W','W','W','W',...
                 'W','W','W','t','e'},'CategoricalPredictors',...
                 'Xvar');

%--------------------------------------------------------------------------
%                                                                         %
%      Problem 4.b, Kaggle Accuracy =78%                                                         %
%                                                                         %
%                                                                         %
%                                                                         %
%--------------------------------------------------------------------------
bagAgeFareTree = predict(AgeTreeBag1,[Age1, Fare1]);

%Messing with bagging code stuff...
%B = TreeBagger(7,[Age,Fare,Embarked,Parch,Pclass,Sex,SibSp]...
 %   ,Survived, 'OOBPred', 'on');



RandomForestPredictions = predict(MdlRandomForest,[Age1, Fare1]);




