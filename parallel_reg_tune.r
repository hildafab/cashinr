library("mlr")
library("OpenML")

library("parallel")
library("foreach")
library("doParallel")

#detectCores() - 1

cl <- makeCluster(2)
registerDoParallel(cl, cores = 2)

regression_task_ids = c(4763,4839,5038,5122,5141,5172,4841,4858,4877,5027,2294,2319,7573,3002,4818)

data = foreach(i = 1:length(regression_task_ids), .packages = c("mlr","OpenML", "mlrMBO", "irace", "nnet","e1071", "randomForest"), .combine = rbind) %dopar% {
                 try({
                   
                   task.id = regression_task_ids[i]
                   
                   budget = 2
                   # SVM classifier
                   lrn1 = makeLearner("regr.svm") #, predict.type="prob")
                   # SVM hyper-parameter search space
                   ps1 = makeParamSet(
                     makeNumericParam("cost", lower=0, upper=15, trafo=function(x) 2^x),
                     makeNumericParam("gamma", lower=0, upper=15, trafo=function(x) 2^x)
                   )
                   
                   
                   # random forest classifier
                   lrn2 = makeLearner("regr.randomForest") #, predict.type="prob")
                   # random forest hyper-parameter search space
                   ps2 = makeParamSet(
                     makeIntegerParam("mtry", lower = 1L, upper = 5000L),
                     makeIntegerParam("ntree", lower=1L, upper=5000L),
                     makeIntegerParam("nodesize", lower = 1L, upper = 5000L )
                   )
                   
                   
                   # neural network classifier
                   lrn3 = makeLearner("regr.nnet") #, predict.type="prob")
                   # neural network hyper-parameter search space
                   ps3 = makeParamSet(
                     makeIntegerParam("maxit", lower = 1, upper = 1000L),
                     makeIntegerParam("size", lower = 0, upper = 500L),
                     makeNumericParam("decay", lower = 0, upper = 100L),
                     makeNumericParam("rang", lower = -10, upper = 10)
                   )
                   
                   #Decision Tree
                   lrn4 = makeLearner("regr.rpart") #, predict.type="prob")
                   # Decision Tree hyper-parameter search space
                   ps4 = makeParamSet(
                     makeIntegerParam("maxdepth", lower = 1, upper = 30),
                     makeIntegerParam("minsplit", lower = 1, upper = 1000L),
                     makeNumericParam("cp", lower = 0, upper = 1),
                     makeIntegerParam("minbucket", lower = 1, upper = 1000L)
                   )
                   
                   # Fast knn classifier
                   #lrn5 = makeLearner("regr.fnn") #, predict.type="prob")
                   # Fast knn hyper-parameter search space
                   #ps5 = makeParamSet(
                   
                   #)
                   
                   # knn classifier
                   #lrn6 = makeLearner("regr.IBk") #) #, predict.type="prob")
                   # Knn hyper-parameter search space
                   #ps6 = makeParamSet(
                   
                   #)
                   
                   all.learners = list(lrn1,lrn2,lrn3,lrn4);
                   
                   #learner = makeModelMultiplexer(base.learners = all.learners)
                   
                   all.parametersets = list(ps1,ps2,ps3,ps4)
                   
                   lrn.ps = list(all.learners,all.parametersets)
                   
                   print(paste("Task ID:",task.id))
                   i=1
                   for(i in 1:length(lrn.ps[[1]])){
                   
                     lrn = lrn.ps[[1]][[i]]
                     ps = lrn.ps[[2]][[i]]
                     
                     print(paste("Learner",toString(class(lrn))))
                     
                     # specify the task
                     oml.task = getOMLTask(task.id)
                     
                     # make a check is imputation is needed
                     if (any(is.na(oml.task$input$data$data))) {
                       catf(" - Data imputation required ...")
                       temp = impute(data = oml.task$input$data.set$data, classes = list(numeric = imputeMean(), factor = imputeMode()))
                       oml.task$input$data.set$data = temp$data
                     }
                     
                     obj = convertOMLTaskToMlr(oml.task)
                     obj$mlr.rin = makeResampleDesc("CV", iters = 10L)
                     
                     # Random Search
                     ctrl.random = makeTuneControlRandom(maxit = budget)
                     
                     # Grid Search 
                     ctrl.grid   = makeTuneControlGrid(resolution=round(sqrt(budget))) # 2 dimensions
                     
                     # List of tuning controls
                     ctrls = list(ctrl.random, ctrl.grid) #, ctrl.irace, ctrl.gensa, ctrl.cmaes)
                     # predict.type = "se"
                     
                     inner = makeResampleDesc("CV", iters=5)
                     outer = makeResampleInstance("CV", iters=10, task=obj$mlr.task)
                     
                     # Calling tuning techniques (for each tuning control ... )
                     aux = lapply(ctrls, function(ct) {
                       
                       print(paste("Control",toString(class(ct))))
                       
                       tuned.learner = makeTuneWrapper(learner=lrn, resampling=inner, par.set=ps, 
                                                       control=ct, show.info=FALSE)
                       
                       res = resample(learner=tuned.learner, task=obj$mlr.task, resampling=outer, 
                                      extract=getTuneResult, models=TRUE, show.info = FALSE, 
                                      measures=list(mse, rmse, timetrain, timepredict, timeboth))
                       
                       return(res)
                     })
                     
                     print(aux)
                     
                     #output = tuningTask(oml.task.id = task.id, learner = lrn, par.set = ps, budget = budget)
                     #print(output)
                     
                     # specify the task
                     oml.task = getOMLTask(task.id)
                     
                     # make a check is imputation is needed
                     if (any(is.na(oml.task$input$data$data))) {
                       catf(" - Data imputation required ...")
                       temp = impute(data = oml.task$input$data.set$data, classes = list(numeric = imputeMean(), factor = imputeMode()))
                       oml.task$input$data.set$data = temp$data
                     }
                     
                     obj = convertOMLTaskToMlr(oml.task)
                     obj$mlr.rin = makeResampleDesc("CV", iters = 10L)
                     
                     bag.learner = makeBaggingWrapper(lrn)
                     bag.learner = setPredictType(bag.learner, predict.type = "se")
                     
                     outer = makeResampleInstance("CV", iters=10, task=obj$mlr.task)
                     
                     # define the objective function (intern function)
                     myObjectiveFunction = function(x){
                       
                       obj = convertOMLTaskToMlr(oml.task)
                       rdesc = makeResampleInstance("CV", iters=5, task=obj$mlr.task)
                       
                       # modifying the learner
                       new.learner = setHyperPars(lrn, par.vals = x)
                       
                       res = resample(learner=new.learner, task=obj$mlr.task, resampling=rdesc, 
                                      models=TRUE, show.info = FALSE, measures=list(mse, rmse, timetrain, timepredict, timeboth))
                       
                       value = res$aggr[1]
                       return(value)
                       
                     }
                     
                     mbo.control = makeMBOControl(iters = budget, init.design.points = 30L)
                     mbo.control = setMBOControlInfill(mbo.control, crit = "ei")
                     
                     print(paste("Control",toString(class(mbo.control))))
                     
                     mbo.result = mbo(myObjectiveFunction, par.set = ps, learner = bag.learner, 
                                      control = mbo.control, show.info = TRUE)
                     tuned.learner = makeBaggingWrapper(makeLearner(cl = toString(class(lrn)[1]),par.vals = mbo.result$x))
                     tuned.learner = setPredictType(tuned.learner, predict.type = "se")
                     
                     res = resample(learner=tuned.learner, task=obj$mlr.task, resampling=outer, 
                                    models=TRUE, show.info = FALSE, 
                                    measures=list(mse, rmse, timetrain, timepredict, timeboth))
                     
                     print(res)
                     
                     #output = mboTuningTask(oml.task.id = task.id, learner = lrn, par.set = ps, budget = budget)
                     #print(output)
                   }
                   
                   
                 })
               }

stopCluster(cl);