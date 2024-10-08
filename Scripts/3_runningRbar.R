library(dplR) #import for chronology building
library(dplyr)


rm(list=ls())
Workdir<-here::here('Data/QWA/detrended')
filelist=list.files(Workdir,pattern = "pbw*", full.names = TRUE)
exclude<-read.csv(here::here("Data/QWA/raw/NA_interpolations.csv"))

file_paths= list.files(Workdir, pattern = "pbw*", full.names = FALSE)



for(i in 1:length(file_paths)){
series=read.rwl(filelist[i])


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
window.length=50
stats=rwi.stats.running(Badremoved, ids = NULL, period = "max",
                        method = "spearman",
                        prewhiten=FALSE,n=NULL,
                        running.window = TRUE,
                        window.length=50,
                        window.overlap = 25,
                        first.start = NULL,
                        min.corr.overlap = min(30, window.length),
                        round.decimals = 3,
                        zero.is.missing = TRUE)

write.csv(stats,paste(here::here('Data/QWA/chronology_stats/'),"rbar50year",file_paths[i],sep=''))

}


