library(dplR)
library(ggplot2)
library(zoo)
library(tidyr)
library(tidyverse)


# Define the base path
base_path <- "~/Documents/Projects/MANCHA/toGit/Firth/Data/QWA/raw/"

# List all files that match the expected pattern
file_paths <- list.files(base_path, pattern = "MXDCWA_.*_.*mu\\.txt", full.names = TRUE)


# Function to process each file
process_file <- function(file_path) {
  # Extract method and resolution from the filename
  parts <- strsplit(basename(file_path), "_")[[1]]
  method <- parts[2]
  resolution <- gsub(".txt", "", parts[3])
  
  # Read data
  data <- read.table(file_path, header = TRUE, row.names = 1,sep=' ')

  # Convert to RWL format
  rwl_data <- as.rwl(data)
  
  # Apply power transformation
  powt_data <- powt(rwl_data, rescale = TRUE)
  original_row_names <- row.names(data)
  
  data <- data.frame(na.approx(data, na.rm = FALSE))
  row.names(data) <- original_row_names
  stats_data=rwl.stats(rwl_data)

  powt_data <- data.frame(na.approx(powt_data, na.rm = FALSE))
  row.names(powt_data) <- original_row_names
  powt_d=as.rwl(powt_data)
  stats_powt=rwl.stats(powt_d)


  # Function to calculate moving stats and correlation
  calc_stats <- function(series) {
    moving_avg <- rollapply(series, 2, mean, fill = NA, align = "right")
    moving_sd <- rollapply(series, 2, sd, fill = NA, align = "right")
   return(cor(moving_avg, moving_sd, use = "complete.obs"))
  }
  
  # Calculate correlations
  cor_before <- sapply(data, calc_stats)
  cor_after <- sapply(powt_data, calc_stats)
  
  return(data.frame(Series = names(cor_before),
                    Correlation_Before = cor_before,
                    Correlation_After = cor_after,
                    Method = method,
                    Resolution = resolution,
                    Resolution_Order = as.numeric(gsub("mu", "", resolution))))

}

# Process all files and compile results
results <- do.call(rbind, lapply(file_paths, process_file))
results_long <- pivot_longer(results, cols = c("Correlation_Before", "Correlation_After"),
                             names_to = "Condition", values_to = "Correlation")
results_long <- results_long %>%
  arrange(Method, Resolution_Order) %>%
  mutate(Resolution = factor(Resolution, levels = unique(Resolution)))

# Plotting the results using bar plots
ggplot(results_long, aes(x = Series, y = Correlation, fill = Condition)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7)) +
  facet_grid(Resolution ~ Method, scales = "free_x", space = "free") +
  labs(x = NULL, y = "Spread versus Level Correlation",
       title = "") +
  theme_minimal() +
  theme(axis.text.x = element_blank(),  # Hide x-axis labels
        axis.ticks.x = element_blank(), # Hide x-axis ticks
        strip.background = element_blank(),
        strip.text.x = element_text(size = 12, face = "bold"),
        strip.text.y = element_text(size = 12, face = "bold"))

ggsave("~/Documents/Projects/MANCHA/toGit/Firth/Figures/SpreadvsLevel.png",bg='white')
