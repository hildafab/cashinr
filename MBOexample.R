#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

library("mlr")
library("OpenML")
library("mlrMBO")

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

tuningTask = function(oml.task.id, learner, par.set, budget) {

  # specify the task
  oml.task = getOMLTask(oml.task.id)

  # make a check is imputation is needed
  if (any(is.na(oml.task$input$data$data))) {
    catf(" - Data imputation required ...")
    temp = impute(data = oml.task$input$data.set$data, classes = list(numeric = imputeMean(), factor = imputeMode()))
    oml.task$input$data.set$data = temp$data
  }

  obj = convertOMLTaskToMlr(oml.task)
  obj$mlr.rin = makeResampleDesc("CV", iters = 10L)

  # Random Search
  ctrl.random = makeTuneControlRandom(maxit=budget)
  # List of tuning controls
  ctrls = list(ctrl.random )# ctrl.grid #, ctrl.irace, ctrl.gensa, ctrl.cmaes)
# predict.type = "se"
 
  inner = makeResampleDesc("CV", iters=5)
  outer = makeResampleInstance("CV", iters=10, task=obj$mlr.task)
    
  # Calling tuning techniques (for each tuning control ... )
  aux = lapply(ctrls, function(ct) {
    
    tuned.learner = makeTuneWrapper(learner=learner, resampling=inner, par.set=par.set, 
      control=ct, show.info=FALSE)
    
    res = resample(learner=tuned.learner, task=obj$mlr.task, resampling=outer, 
      extract=getTuneResult, models=TRUE, show.info = FALSE, 
      measures=list(mse, rmse, timetrain, timepredict, timeboth))

    return(res)
  })

  return(aux)
}

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

MBOTuning = function(oml.task.id, learner, par.set, budget = NULL) {

  mbo.control = makeMBOControl(iters = 10L, init.design.points = 30L)
  mbo.control = setMBOControlInfill(mbo.control, crit = "ei")

  oml.task = getOMLTask(oml.task.id)

  # make a check is imputation is needed
  if (any(is.na(oml.task$input$data$data))) {
    catf("   - Data imputation required ...")
    temp = impute(data = oml.task$input$data.set$data, classes = list(numeric = imputeMean(), factor = imputeMode()))
    oml.task$input$data.set$data = temp$data
  }

  # define the objective function (intern function)
  myObjectiveFunction = function(x){

    obj = convertOMLTaskToMlr(oml.task)
    rdesc = makeResampleInstance("CV", iters=5, task=obj$mlr.task)
   
    # modifying the learner
    new.learner = setHyperPars(learner, par.vals = list(cost = x[[1]], 
      gamma = x[[2]]))

    res = resample(learner=new.learner, task=obj$mlr.task, resampling=rdesc, 
     models=TRUE, show.info = FALSE, measures=list(mse, rmse, timetrain, timepredict, timeboth))

    value = res$aggr[1]
    return(value)

  }

  mbo.result = mbo(myObjectiveFunction, par.set = par.set, learner = learner, 
    control = mbo.control, show.info = TRUE)

  return(mbo.result)

}

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

main = function() {

  task.id = 5141
  budget = 20

  # SVM regrersors
  lrn = makeLearner("regr.svm")

  # SVM hyper-parameter search space
  ps = makeParamSet(
    makeNumericParam("cost", lower=-15, upper=15, trafo=function(x) 2^x),
    makeNumericParam("gamma", lower=-15, upper=15, trafo=function(x) 2^x)
  )

  # output = tuningTask(oml.task.id = task.id, learner = lrn, par.set = ps, budget = 20)
  # print(output)

  # List of mlr Regressors (see those with 'se option in the predicted type) 
  #  https://mlr-org.github.io/mlr-tutorial/devel/html/integrated_learners/index.html

#  random forest regressor
  # lrn2 = makeLearner("regr.randomForest", predict.type = "se")
#   lrn2 = makeLearner("regr.randomForest") #, predict.type="prob")
#   # random forest hyper-parameter search space
#   ps2 = makeParamSet(
#     makeIntegerParam("ntree", lower=1L, upper=500L),
#     makeIntegerParam("nodesize", lower = 1L, upper = 100L )
#   )

  catf(" * MBO optimization")
  bag.lrn = makeBaggingWrapper(lrn)
  lrn2 = setPredictType(bag.lrn, predict.type = "se")

  mbo.output = MBOTuning(oml.task.id = task.id, learner = lrn2, par.set = ps)
  print(mbo.output)

}


#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

main()

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
