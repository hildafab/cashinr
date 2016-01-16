
library("mlr")
library("OpenML")

saveOMLConfig(apikey = "dce6d7b81d7eb26de554be95c812f0db", overwrite = TRUE)

dataset = getOMLDataSet(did = 37)

dataset$data

# SVM classifier
lrn = makeLearner("classif.svm", predict.type="prob")

# SVM hyper-parameter search space
ps = makeParamSet(
  makeNumericParam("cost", lower=-15, upper=15, trafo=function(x) 2^x),
  makeNumericParam("gamma", lower=-15, upper=15, trafo=function(x) 2^x)
)

# specify the task
openml.task = getOMLTask(task.id = 37)
obj = convertOMLTaskToMlr(openml.task)

# mlr.task = makeClassifTask(data = dataset$data, target = "class")
# obj$mlr.task 
# rdesc = makeResampleDesc("CV", iters = 10, stratify = TRUE)
# obj$mlr.rin

run = resample(learner= lrn, task=obj$mlr.task, resampling = obj$mlr.rin, models = TRUE, measures=list(acc, ber))

# random forest classifier
lrn2 = makeLearner("classif.randomForest", predict.type="prob")

# SVM hyper-parameter search space
ps2 = makeParamSet(
  makeIntegerParam("ntree", lower=1L, upper=500L)
)

learners = makeModelMultiplexer(base.learners = list(lrn,lrn2))

paramsets = makeModelMultiplexerParamSet(learners, classif.svm = ps, classif.randomForest = ps2)

resample.desc = makeResampleDesc("CV", iters = 10L)
# to save some time we use random search. but you probably want something like this:
# ctrl = makeTuneControlIrace(maxExperiments = 500L)

# 120 evaluations (just fot test)
BUDGET = 120

# Tuning control strategies
# Grid Search 
ctrl.grid   = makeTuneControlGrid(resolution=round(sqrt(BUDGET))) # 2 dimensions

#Random Search
ctrl.random = makeTuneControlRandom(maxit=BUDGET)

ctrl.irace  = makeTuneControlIrace(maxExperiments = BUDGET)

# List of tuning controls
ctrls = list(ctrl.grid, ctrl.random , ctrl.irace)

#res = tuneParams(lrn, iris.task, rdesc, par.set = ps, control = ctrl)
#print(res)
# Calling tuning techniques (for each tuning control ... )

openml.classif.task = getOMLTask(task.id = 37)

mlr.classif.task = convertOMLTaskToMlr(openml.classif.task)

aux = lapply(ctrls, function(ct) {
  
  lrns = makeTuneWrapper(learner=learners, resampling=resample.desc, par.set=paramsets, control=ct, show.info=FALSE)
  
  res = resample(learner=lrns, task=mlr.classif.task$mlr.task, extract=getTuneResult, resampling=resample.desc,
                 models=TRUE, show.info = FALSE, measures=list(acc)
  )
  
  return(res)
})

# Retuning a list with results
return(aux)

#git remote add origin git@github.com:hildafab/cashinr.git
#git push -u origin master

