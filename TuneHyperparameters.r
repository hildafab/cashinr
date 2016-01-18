

TuneTask = function(openML.task, learner, paramsets, ctrls, result.measures){
  
  library("mlr")
  library("OpenML")
  
  saveOMLConfig(apikey = "dce6d7b81d7eb26de554be95c812f0db", overwrite = TRUE)
  
  resample.desc = makeResampleDesc("CV", iters = 10L)
  
  mlr.classif.task = convertOMLTaskToMlr(openML.task)
  
  result = lapply(ctrls, function(ct) {
    
    lrns = makeTuneWrapper(learner=learner, resampling=resample.desc, par.set=paramsets, control=ct, show.info=FALSE)
    
    res = resample(learner=lrns, task=mlr.classif.task$mlr.task, extract=getTuneResult, resampling=resample.desc,
             models=TRUE, show.info = FALSE, measures=result.measures)
    
    save(res, file = mlr.classif.task$mlr.task$task.desc$id+class(res)[1])
    
    return(res)
    })
  
  return(result)
}


TuneTaskFromData = function(openml.dataset, learner, paramsets, ctrls, result.measures){
  
  library("mlr")
  library("OpenML")
  
  saveOMLConfig(apikey = "dce6d7b81d7eb26de554be95c812f0db", overwrite = TRUE)
  
  resample.desc = makeResampleDesc("CV", iters = 10L)
  
  #data.no.missing = openml.dataset$data[complete.cases(openml.dataset$data),]
  
  imp = impute(openml.dataset$data, classes = list(integer = imputeMean(), factor = imputeMode()),
               dummy.classes = "integer")
  
  mlr.classif.task = makeClassifTask(data = imp$data, target = openml.dataset$desc$default.target.attribute)
  
  result = lapply(ctrls, function(ct) {
    
    lrns = makeTuneWrapper(learner=learner, resampling=resample.desc, par.set=paramsets, control=ct, show.info=FALSE)
    
    res = resample(learner=lrns, task=mlr.classif.task, extract=getTuneResult, resampling=resample.desc,
                   models=TRUE, show.info = FALSE, measures=result.measures)
    
    save(res, file = paste(mlr.classif.task$mlr.task$task.desc$id,class(res)[1],sep = ""))
    
    return(res)
  })
  
  return(result)
}

