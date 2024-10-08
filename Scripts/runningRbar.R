library(dplR) #import for chronology building
library(dplyr)
rm(list=ls())
Workdir='/Users/julieedwards/Documents/Projects/MANCHA/MXD/nomcrb08_June2024/'
setwd(Workdir)
filelist=list.files(Workdir)
exclude=read.csv("/Users/julieedwards/Documents/Projects/MANCHA/QWAData/FINAL/concat/Summary/PoI/NA_interpolations.csv")

file='pbw80.ind'


series=read.rwl(file)


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
window.length=100
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
plot(stats$mid.year,stats$rbar.eff,type='s',cex.main = .5,ylim = c(0, 0.8),xlab = "Year", ylab = "rbar")
title(main= "50-year moving window rbar: pbw20")
grid()

