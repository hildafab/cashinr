
#Get learners and parameter sets
getLearnersAndParamSets = function(){
# SVM classifier
lrn1 = makeLearner("classif.svm", predict.type="prob")
# SVM hyper-parameter search space
ps1 = makeParamSet(
  makeNumericParam("cost", lower=-15, upper=15, trafo=function(x) 2^x),
  makeNumericParam("gamma", lower=-15, upper=15, trafo=function(x) 2^x)
)


# random forest classifier
lrn2 = makeLearner("classif.randomForest", predict.type="prob")
# random forest hyper-parameter search space
ps2 = makeParamSet(
  makeIntegerParam("ntree", lower=1L, upper=500L),
  makeIntegerParam("nodesize", lower = 1L, upper = 100L )
)


# neural network classifier
lrn3 = makeLearner("classif.nnet", predict.type="prob")
# neural network hyper-parameter search space
ps3 = makeParamSet(
  makeIntegerParam("maxit", lower = 1, upper = 500)#,
  #makeIntegerParam("size", lower = 0, upper = 10)
)

#Decision Tree
lrn4 = makeLearner("classif.rpart", predict.type="prob")
# Decision Tree hyper-parameter search space
ps4 = makeParamSet(
  makeIntegerParam("maxdepth", lower = 1, upper = 30),
  makeIntegerParam("minsplit", lower = 1, upper = 500)
)

# Fast knn classifier
#lrn5 = makeLearner("classif.fnn", predict.type="prob")
# Fast knn hyper-parameter search space
#ps5 = makeParamSet(

#)

# knn classifier
#lrn6 = makeLearner("classif.IBk") #, predict.type="prob")
# Knn hyper-parameter search space
#ps6 = makeParamSet(

#)

all.learners = list(lrn1,lrn2,lrn3,lrn4);

learner = makeModelMultiplexer(base.learners = all.learners)

all.parametersets = c(ps1,ps2,ps3,ps4)
#c(classif.svm = ps1, classif.randomForest = ps2, classif.avNNet=ps3, 
#classif.rpart=ps4)


paramsets = makeModelMultiplexerParamSet(learner, classif.svm = ps1, classif.randomForest = ps2, classif.nnet=ps3, 
                                         classif.rpart=ps4)

return(list(learner, paramsets));
}

#Get controls
getControls = function(){

# Tuning control strategies

# 250 evaluations 
BUDGET = 250

# Grid Search 
ctrl.grid   = makeTuneControlGrid(resolution=round(sqrt(BUDGET))) # 2 dimensions

#Random Search
ctrl.random = makeTuneControlRandom(maxit=BUDGET)

#IRace
ctrl.irace  = makeTuneControlIrace(maxExperiments = BUDGET)

#GenSA
ctrl.gensa = makeTuneControlGenSA(budget = BUDGET)

#CMAES
ctrl.cmaes = makeTuneControlCMAES(budget = BUDGET)

#To Add - nn and pso

# List of tuning controls
ctrls = list(ctrl.grid, ctrl.random , ctrl.irace, ctrl.gensa, ctrl.cmaes)

return(ctrls)
}

#Result measures

getResultMeasures = function(){
result.measures = list(acc,auc,timepredict,timetrain,timeboth,ber)
}


#get data
getData = function(){

TaskIds = c(3709,1966,4571,3667,3863,1970,3826,3663,#10089,
            9961,4201,37,10,9940,9947,3644,9969,4275,35,3748,284) #,8)
DatasetIds = c(844,27,1003,802,1000,34,963,798,#1455,
               1498,337,37,10,1524,1512,778,1506,481,35,885,55) #,8)

dataset.names = list();

i=1;

for (dataset.id in DatasetIds) {
  openml.dataset = getOMLDataSet(did = dataset.id)
  dataset.names[i]=openml.dataset$desc$name
  i=i+1
}

#names(TaskIds) = dataset.names
#names(DatasetIds) = dataset.names

return(list(DatasetIds,TaskIds,dataset.names))

}










