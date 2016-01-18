
#Get all input parameters

Data.from.openMl = getData();

DatasetIds = Data.from.openMl[[1]]
TaskIds = Data.from.openMl[[2]]
dataset.names = Data.from.openMl[3]

learner.and.paramset = getLearnersAndParamSets();

learner = learner.and.paramset[1]
paramsets = learner.and.paramset[2]

tune.controls = getControls()

result.measures = getResultMeasures()


# Tune based on Tasks
i=1
for(taskid in TaskIds){
  openml.task = getOMLTask(task.id = taskid)
  dataset.name = dataset.names[i]
  dataset.result = TuneTask(openML.task = openml.task, learner = learner, paramsets = paramsets, ctrls = tune.controls, 
                            result.measures = result.measures)
  save(dataset.result,file = dataset.name)
}


#Tune Based on Data

for(i in 1:NROW(DatasetIds)){
  dataset.id = DatasetIds[[i]][[1]]
  openml.dataset = getOMLDataSet(did = dataset.id)
  result = TuneTaskFromData(openml.dataset = openml.dataset, learner = learner[[1]], paramsets = paramsets[[1]], ctrls = tune.controls, 
                   result.measures = result.measures)
  dataset.name = dataset.names[i]
  save(result,file = dataset.name)
  #i=i+1
}