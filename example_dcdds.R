library(RANN)
library(UniDOE)
library(SPlit)
library(ggplot2)
library(numbers)
library(knn.covertree)
source("DC_DDS.R")

# Your feature path.
path = "feature/resnet50_moco_cifar10_feature.txt"
# b: number of block, m: subset in one block, b * m = n.
# seed: random seed, dim: PCA reduced dimension.
m = 25
b = 20
dim = 5
seed = 100

data = read.table(path)
selected_data = seq_DDS_givendim(data, m, b, dim, seed)
write.csv(selected_data, paste("cifar", dim, m, b, "indices.csv", sep = "_"),row.names = F)
