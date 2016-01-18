
#Get datasets and tasks to perform tuning.

library("mlr")
library("OpenML")

saveOMLConfig(apikey = "dce6d7b81d7eb26de554be95c812f0db", overwrite = TRUE)

TaskIds = c(3709,1966,4571,3667,3863,1970,3826,3663,10089,9961,4201,37,10,9940,9947,3644,9969,4275,35,3748,284,8)
DatasetIds = c(844,27,1003,802,1000,34,963,798,1455,1498,337,37,10,1524,1512,778,1506,481,35,885,55,8)

dataset.names = list();

i=1;

for (dataset.id in DatasetIds) {
  openml.dataset = getOMLDataSet(did = dataset.id)
  dataset.names[i]=openml.dataset$desc$name
  i=i+1
}

names(TaskIds) = dataset.names
names(DatasetIds) = dataset.names



