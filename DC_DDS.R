library(RANN)
library(UniDOE)
library(SPlit)
library(ggplot2)
library(numbers)
library(knn.covertree)
library(MVN)

GLP<-function(n,p,type="CD2"){
  fb<-c(3,5,8,13,21,34,55,89,144,233,377,610,987)
  if(((n+1)%in%fb)&(p==2)){
    design0<-matrix(0,(n+1),p)
    H<-rep(1,2)
    H[2]<-fb[which(fb==(n+1))-1]
    for (j in 1:p) {
      for (i in 1:(n+1)) {
        design0[i,j]<-(2*i*H[j]-1)/(2*(n+1))-floor((2*i*H[j]-1)/(2*(n+1)))
      }
    }
    design0<-design0[-(n+1),]*(n+1)/n
  }else{
    if(p==1){
      design0<-matrix(0,n,p)
      for(i in 1:n){
        design0[i,1]<-(2*i-1)/(2*n)
      }
      return(design0)
    }
    h<-c()
    for(i in 2:min((n+1),200)){
      if(coprime((n+1),i)==T){
        h<-c(h,i)
      }
    }
    if(p>2){
      for (i in 1:100) {
        if(choose(p+i,i)>5000){
          addnumber<-i
          break
        }
      }
      h<-h[sample(1:length(h),min(length(h),(p+addnumber)))]
    }
    H<-combn(h,p,simplify = F)
    if(length(H)>3000){
      H<-H[sample(3000)]
    }
    design0<-matrix(0,n,p)
    d0<-DesignEval(design0,crit=type)
    for (t in 1:length(H)) {
      design<-matrix(0,n,p)
      for (i in 1:p) {
        for (j in 1:n) {
          design[j,i]<-(j*H[[t]][i])%%(n+1)
        }
      }
      d1<-DesignEval(design,crit=type)
      if(d1<d0){
        d0<-d1
        design0<-design
      }
    }
    design0<-(design0*2-1)/(2*n)
  }
  return(design0)
}

BRUD<-function(Design){
  D=Design
  n=nrow(D)
  s=ncol(D)
  rand=matrix(runif((n*s),0,1),nrow=n,ncol=s)
  eta_mat=matrix(rep(sample(c(0:(n-1)),s,replace = TRUE),n),nrow=n,ncol=s,byrow=TRUE)
  eta_mat=((eta_mat-0.5)/n)
  RUD=(D+eta_mat)%%1+rand/n
  #RUD=1-abs(2*RUD-1)
  return(RUD)
}

DDS<-function(data,n,type="CD2",Design=NULL,ratio=NULL,reduced.dim=NULL,scale=FALSE){
  if((type %in% c("CD2","WD2","MD2"))==F){
    stop("type shoud be chose from 'CD2','WD2','MD2'")
  }
  if(n>dim(data)[1]){
    stop("The subsample size n should be less than sample size N")
  }
  if(scale==TRUE){
    x<-scale(as.matrix(data),scale=TRUE)
  }
  if(scale==FALSE){
    x<-scale(as.matrix(data),scale=FALSE)
  }
  if(is.null(ratio)&&is.null(reduced.dim)){
    stop("Proportion of cumulative explained variance or manually set reduced dimension is indispensable for PCA dimension reduction")
  }
  if(is.numeric(ratio)&&is.numeric(reduced.dim)){
    stop("PCA takes either proportion of cumulative explained variance or reduced dimension as basis to reduce dimension")
  }
  y<-n
  N<-dim(x)[1]
  p0<-dim(x)[2]
  svddata<-svd(x)
  if(is.numeric(ratio)){
    lambda<-(svddata$d)^2
    sumlambda<-sum(lambda)
    for (i in 2:p0) {
      if(sum(lambda[1:i])>=(ratio*sumlambda)){
        p<-i
        print(paste("Take ratio =",ratio,",","p after pca =",i))
        break
      }
    }
  }
  if(is.numeric(reduced.dim)){
    p=reduced.dim
  }
  rdata<-x%*%svddata$v[,1:p]
  if(is.null(Design)){
    design<-GLP(n,p,type)
  }
  else{
    design=BRUD(Design)
  }
  yita<-matrix(nrow=n,ncol=p)
  for (i in 1:p) {
    for (j in 1:n) {
      yita[j,i]<-quantile(rdata[,i],design[j,i])
    }
  }
  kdtree<-nn2(data=rdata,query=yita,k=1,treetype="kd")
  subsample<-kdtree$nn.idx
  return(subsample)
}

CVgroup <- function(k,datasize,seed){
  cvlist <- list()
  set.seed(seed)
  n <- rep(1:k,ceiling(datasize/k))[1:datasize]    
  temp <- sample(n,datasize)   
  x <- 1:k
  dataseq <- 1:datasize
  cvlist <- lapply(x,function(x) dataseq[temp==x])  
}

seq_DDS_givendim<-function(data,n,k,reduced.dim,seed=100, Design=NULL,type="CD2",ratio=NULL,scale=FALSE){
  if((type %in% c("CD2","WD2","MD2"))==F){
    stop("type shoud be chose from 'CD2','WD2','MD2'")
  }
  if(n>dim(data)[1]){
    stop("The subsample size n should be less than sample size N")
  }
  td<-reduced.dim
  if(is.null(Design)){
    design=GLP(n,td,type)
  }
  else{
    design=Design
  }
  indices<-array()
  data_collection<-list()
  un_sampled_collection<-list()
  N<-dim(data)[1]
  indices_collection<-CVgroup(k,N,seed)
  for(i in 1:k){
    data_collection[[i]]=data[indices_collection[[i]],]
    un_sampled_collection[[i]]=DDS(data=data_collection[[i]],n=n,reduced.dim=td,Design=design,type=type,ratio=ratio,scale=scale)[,1]
    sampled_indices=indices_collection[[i]][un_sampled_collection[[i]]]
    indices=c(indices,sampled_indices)
  }
  indices=indices[-1]
  return(indices)
}

