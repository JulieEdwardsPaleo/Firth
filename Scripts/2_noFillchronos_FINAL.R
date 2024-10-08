library(dplR) #import for chronology building
library(dplyr)
rm(list=ls())
Workdir='/Users/julieedwards/Documents/Projects/MANCHA/MXD/nomcrb08_June2024/'
setwd(Workdir)
filelist=list.files(Workdir)
exclude=read.csv("/Users/julieedwards/Documents/Projects/MANCHA/QWAData/FINAL/concat/Summary/PoI/NA_interpolations.csv")


for(i in 1:length(filelist)){
  
  series=read.rwl(filelist[i])
  write.csv(summary(series),paste(Workdir,"RWL_summary_",filelist[i],sep='')) # write summary stats of series
  
  if ("MCRB08" %in% colnames(series)) {
    Badremoved=select(series,-MCRB08)
  } else {
    Badremoved=series }
  for (j in 1:nrow(exclude)) {
    woodid <- exclude$SeriesID[j]
    year <- as.character(exclude$Year[j])
    
    # Check if the current 'WOODID' and 'year' exist in 'series' DataFrame
    if (woodid %in% colnames(series) && year %in% rownames(series)) {
      # Set the matching cell value to NaN
      Badremoved[year, woodid] <- NA
    }
  }
  
  write.csv(rwi.stats(Badremoved,period = 'max'),paste(Workdir,"RWI_stats",filelist[i],sep=''))
  
  Chrono=chron(Badremoved,biweight = TRUE)
  write.csv(Chrono,paste(Workdir,"noFill_",filelist[i],sep=''))
  
  
  
}
