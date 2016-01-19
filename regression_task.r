library(soobench)
library(mlr)
library(mlrMBO)
library(ParamHelpers)

regression_task_ids = c(5146,4763,4839,5038,5122,5141,5172,4841,4858,4877,5027,2294,2319,7573,3002,4818)


reg_task_id = regression_task_ids[1]

reg.task = getOMLTask(task.id = reg_task_id)

mlr.task = convertOMLTaskToMlr(reg.task)

learner.and.paramset = getregrLearnersAndParamSets();

learner = learner.and.paramset[[1]]
paramsets = learner.and.paramset[[2]]

# SVM classifier
reg.lrn1 = makeLearner("regr.svm") #, predict.type="prob")

bag.lrn1 = makeBaggingWrapper(reg.lrn1)

bag.lrn1 = setPredictType(lrn1, predict.type = "se")

# SVM hyper-parameter search space
ps1 = makeParamSet(
  makeNumericParam("cost", lower=0, upper=15, trafo=function(x) 2^x),
  makeNumericParam("gamma", lower=-15, upper=15, trafo=function(x) 2^x)
)

obj.fun = rastrigin_function(1)
control = makeMBOControl(iters = 10L, init.design.points = 30L)
control = setMBOControlInfill(control, crit = "ei")
hyperparams.mbo = mbo(makeMBOFunction(obj.fun), par.set = ps1, learner = bag.lrn1, control = control, show.info = TRUE)

learn = makeLearner("regr.svm", par.vals = hyperparams.mbo$x)

bag.learn = makeBaggingWrapper(learn)
bag.learn = setPredictType(bag.learn, predict.type = "se")

resample.desc = makeResampleDesc("CV", iters = 10L)

#lrns = makeTuneWrapper(learner=reg.lrn1, resampling=resample.desc, par.set=hyperparams.mbo$x, control=ct, show.info=FALSE)

mlr.task$mlr.task$env$data = mlr.task$mlr.task$env$data[complete.cases(mlr.task$mlr.task$env$data),]
  
  #impute(mlr.task$mlr.task$env$data, classes = list(integer = imputeMean(), factor = imputeMode(), 
                                                                               #numeric = imputeMean()), dummy.classes = "integer")$data

mlr.task.own = makeRegrTask(data = mlr.task$mlr.task$env$data, target = mlr.task$mlr.task$task.desc$target)


res = resample(learner=bag.learn, task=mlr.task.own, resampling=resample.desc,
               models=TRUE, show.info = FALSE, measures=list(timepredict,timetrain,timeboth,rmse))


