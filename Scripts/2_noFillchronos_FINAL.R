library(dplR) #import for chronology building
library(dplyr)
library(here)
rm(list=ls())


# Define the base path
Workdir <- here::here("Data/QWA/detrended/")

filelist=list.files(Workdir)
exclude<-read.csv(here::here("Data/QWA/raw/NA_interpolations.csv"))


for(i in 1:length(filelist)){
  
  series=read.rwl(paste(Workdir,filelist[i],sep=''))
  write.csv(summary(series),paste(here::here('Data/QWA/chronology_stats/'),"RWL_summary_",filelist[i],sep='')) # write summary stats of series
  
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
  
  write.csv(rwi.stats(Badremoved,period = 'max'),paste(here::here('Data/QWA/chronology_stats/'),"RWI_stats",filelist[i],sep=''))
  
  Chrono=chron(Badremoved,biweight = TRUE)
  write.csv(Chrono,paste(here::here('Data/QWA/chronologies/'),"noFill_",filelist[i],sep=''))
  
  
  
}
