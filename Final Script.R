# ================================================================
# Impact of COVID-19 on Corporate Dividend Policy
# ================================================================

# Clear workspace
rm(list = ls())

# Load  libraries
library(tidyverse)
library(fixest)      # For fixed effects models
library(plm)         # Panel data models
library(lmtest)      # Diagnostic tests
library(sandwich)    # Robust standard errors
library(stargazer)   # Output tables
library(ggplot2)     # Visualization
library(car)         # VIF test

# ================================================================
# 1. LOAD AND EXPLORE DATA
# ================================================================

# Load the dataset
df <- read.csv("covid_dividend_macro_firm_controls.csv")

# Ensure Year and Quarter are numeric (not factors)
df$Year <- as.numeric(as.character(df$Year))
df$Quarter <- as.numeric(as.character(df$Quarter))

# Create time variable
df$Time <- df$Year + (df$Quarter - 1) / 4
df$TimeID <- as.numeric(factor(paste(df$Year, df$Quarter, sep = "_")))

# Convert to panel data AFTER creating time variables
df_panel <- pdata.frame(df, index = c("Ticker", "TimeID"))


# Check for missing values
print(colSums(is.na(df)))

# ================================================================
# 2. DESCRIPTIVE STATISTICS
# ================================================================


desc_stats_period <- df %>%
  group_by(Treatment, Post) %>%
  summarise(
    N = n(),
    Mean_Dividends = mean(Total_Dividends, na.rm = TRUE),
    SD_Dividends = sd(Total_Dividends, na.rm = TRUE),
    Median_Dividends = median(Total_Dividends, na.rm = TRUE),
    .groups = 'drop'
  )
print(desc_stats_period)


# ================================================================
# 3. DIFFERENCE-IN-DIFFERENCES ESTIMATION
# ================================================================


# Model 1: Basic DID without controls
did_basic <- lm(Total_Dividends ~ Treatment + Post + Treat_Post, data = df)
summary(did_basic)

# Model 2: DID with macro controls
did_macro <- lm(Total_Dividends ~ Treatment + Post + Treat_Post + 
                  FEDFUNDS + UNRATE + INDPRO + CPIAUCSL, 
                data = df)
summary(did_macro)

# Model 3: DID with firm-level controls
did_controls <- lm(Total_Dividends ~ Treatment + Post + Treat_Post + 
                     FEDFUNDS + UNRATE + INDPRO + CPIAUCSL +
                     log(MarketCap + 1) + PERatio + PB_Ratio + Beta, 
                   data = df)
summary(did_controls)

# Model 4: DID with firm fixed effects
did_fe <- feols(Total_Dividends ~ Treat_Post + FEDFUNDS + UNRATE + 
                  INDPRO + CPIAUCSL + log(MarketCap + 1) + 
                  PERatio + PB_Ratio + Beta | Ticker, 
                data = df, cluster = ~Ticker)
summary(did_fe)

# Model 5: DID with firm and time fixed effects (TWFE)
did_twfe <- feols(Total_Dividends ~ Treat_Post + FEDFUNDS + UNRATE + 
                    INDPRO + log(MarketCap + 1) + PERatio + 
                    PB_Ratio + Beta | Ticker + TimeID, 
                  data = df, cluster = ~Ticker)
summary(did_twfe)

# ================================================================
# 4. ROBUSTNESS CHECKS
# ================================================================

#    ROBUSTNESS CHECK 1: ALTERNATIVE OUTCOME VARIABLE

# Create dividend dummy (1 if dividends > 0)
df$Div_Dummy <- ifelse(df$Total_Dividends > 0, 1, 0)

# Logit model for dividend decision
robust1 <- glm(Div_Dummy ~ Treatment + Post + Treat_Post + 
                 FEDFUNDS + UNRATE + INDPRO + CPIAUCSL +
                 log(MarketCap + 1) + PERatio + PB_Ratio + Beta,
               data = df, family = binomial(link = "logit"))
summary(robust1)

#  ROBUSTNESS CHECK 2: EXCLUDING SPECIFIC QUARTERS\n")


# Exclude Q1 2020 (immediate shock period)
df_no_q1 <- df %>% filter(!(Year == 2020 & Quarter == 1))

robust2 <- feols(Total_Dividends ~ Treat_Post + FEDFUNDS + UNRATE + 
                   INDPRO + log(MarketCap + 1) + PERatio + 
                   PB_Ratio + Beta | Ticker + TimeID,
                 data = df_no_q1, cluster = ~Ticker)
summary(robust2)


#  ROBUSTNESS CHECK 3: PLACEBO TEST (FALSE TREATMENT DATE)


# Create placebo treatment at 2019 Q1
df$Post_Placebo <- ifelse(df$Year >= 2019 & df$Quarter >= 1, 1, 0)
df$Treat_Post_Placebo <- df$Treatment * df$Post_Placebo

# Estimate on pre-COVID period only
df_pre_covid <- df %>% filter(Year < 2020)

placebo <- lm(Total_Dividends ~ Treatment + Post_Placebo + Treat_Post_Placebo + 
                FEDFUNDS + UNRATE + INDPRO + CPIAUCSL,
              data = df_pre_covid)
summary(placebo)


# ================================================================
# 5. DIAGNOSTIC TESTS: PARALLEL TRENDS TEST (PRE-TREATMENT)
# ================================================================

# Calculate mean dividends by group and time
trend_data <- df %>%
  group_by(Treatment, Year, Quarter) %>%
  summarise(Mean_Dividends = mean(Total_Dividends, na.rm = TRUE),
            .groups = 'drop') %>%
  mutate(Time = Year + (Quarter - 1) / 4,
         Group = ifelse(Treatment == 1, "Treatment (COVID-affected)", "Control (Less-affected)"))

# Create plot
p1 <- ggplot(trend_data, aes(x = Time, y = Mean_Dividends, color = Group, linetype = Group)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  geom_vline(xintercept = 2020, linetype = "dashed", color = "red", size = 1) +
  annotate("text", x = 2020, y = max(trend_data$Mean_Dividends, na.rm = TRUE), 
           label = "COVID-19", vjust = -0.5, color = "red") +
  labs(title = "Parallel Trends: Average Dividends Over Time",
       subtitle = "Treatment vs Control Groups (2018-2022)",
       x = "Year",
       y = "Average Total Dividends",
       color = "Group",
       linetype = "Group") +
  theme_minimal() +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold", size = 14),
        axis.title = element_text(face = "bold"))

ggsave("parallel_trends_plot.png", p1, width = 10, height = 6, dpi = 300)
cat("Parallel trends plot saved as 'parallel_trends_plot.png'\n")

# Create time dummies for pre-treatment period
df_pre <- df %>% filter(Post == 0)
df_pre$Time_Treat <- df_pre$Treatment * df_pre$TimeID

pt_test <- lm(Total_Dividends ~ Treatment + factor(TimeID) + 
                factor(TimeID):Treatment,
              data = df_pre)

cat("\nTesting if treatment*time interactions are jointly zero:\n")
pt_coefs <- grep("Treatment:factor\\(TimeID\\)", names(coef(pt_test)), value = TRUE)
pt_test_result <- linearHypothesis(pt_test, pt_coefs)
print(pt_test_result)

if(pt_test_result$`Pr(>F)`[2] > 0.05) {
  cat("\nParallel trends assumption SATISFIED (p > 0.05)\n")
} else {
  cat("\nWARNING: Parallel trends assumption may be VIOLATED (p < 0.05)\n")
}


#  EVENT STUDY (DYNAMIC DID)

# Create relative time variable (quarters relative to treatment)
df$RelTime <- ifelse(df$Post == 0, 
                     df$TimeID - min(df$TimeID[df$Post == 1]),
                     df$TimeID - min(df$TimeID[df$Post == 1]))

# Create event time dummies interacted with treatment
df$RelTime <- as.factor(df$RelTime)

# Event study regression (excluding t=-1 as reference)
event_study <- feols(Total_Dividends ~ i(RelTime, Treatment, ref = -1) + 
                       FEDFUNDS + UNRATE + INDPRO + 
                       log(MarketCap + 1) | Ticker + TimeID,
                     data = df, cluster = ~Ticker)
summary(event_study)

# Plot event study coefficients

png("event_study.png", width = 7, height = 5, units = "in", res = 300)
iplot(event_study, main = "Event Study: Effect of COVID-19 on Dividends")
dev.off()

# ================================================================
# 6. CREATE OUTPUT TABLES
# ================================================================

# Table 1: Main results
stargazer(did_basic, did_macro, did_controls, 
          type = "text",
          title = "DID Estimates",
          dep.var.labels = "Total Dividends")

# Table 2: Fixed effects models
etable(did_fe, did_twfe,
       title = "Table 2: Fixed Effects Models",
       file = "table2_fixed_effects.txt")

# Table 3: Robustness checks
stargazer(robust1, robust2, placebo,
          type = "text",
          title = "Table 3: Robustness Checks",
          column.labels = c("Logit", "Excl Q1-2020", "Placebo"),
          out = "table3_robustness.txt")

# ================================================================
# 7. ECONOMIC SIGNIFICANCE
# ================================================================

# Extract treatment effect from main model
treat_effect <- coef(did_controls)["Treat_Post"]
treat_se <- sqrt(diag(vcov(did_controls)))["Treat_Post"]

# Calculate percentage change
pre_mean_treatment <- mean(df$Total_Dividends[df$Treatment == 1 & df$Post == 0], na.rm = TRUE)
pct_change <- (treat_effect / pre_mean_treatment) * 100

cat("\nTreatment Effect (β₃):", round(treat_effect, 4), "\n")
cat("Standard Error:", round(treat_se, 4), "\n")
cat("Pre-treatment mean (Treatment group):", round(pre_mean_treatment, 4), "\n")
cat("Percentage change:", round(pct_change, 2), "%\n")

# Calculate dollar impact
mean_market_cap <- mean(df$MarketCap[df$Treatment == 1], na.rm = TRUE)
quarterly_impact <- treat_effect * mean_market_cap / 1e9
annual_impact <- quarterly_impact * 4

cat("\nEstimated quarterly dividend reduction:", round(quarterly_impact, 2), "billion USD\n")
cat("Estimated annual dividend reduction:", round(annual_impact, 2), "billion USD\n")

# ================================================================
# 8. SUMMARY OF FINDINGS
# ================================================================


summary_stats <- data.frame(
  Model = c("Basic DID", "Macro Controls", "Full Controls", "Firm FE", "Two-Way FE"),
  Coefficient = c(
    coef(did_basic)["Treat_Post"],
    coef(did_macro)["Treat_Post"],
    coef(did_controls)["Treat_Post"],
    coef(did_fe)["Treat_Post"],
    coef(did_twfe)["Treat_Post"]
  ),
  Std_Error = c(
    sqrt(diag(vcov(did_basic)))["Treat_Post"],
    sqrt(diag(vcov(did_macro)))["Treat_Post"],
    sqrt(diag(vcov(did_controls)))["Treat_Post"],
    se(did_fe)["Treat_Post"],
    se(did_twfe)["Treat_Post"]
  )
)

summary_stats$t_stat <- summary_stats$Coefficient / summary_stats$Std_Error
summary_stats$p_value <- 2 * pt(-abs(summary_stats$t_stat), df = nrow(df) - 5)
summary_stats$Significant <- ifelse(summary_stats$p_value < 0.01, "***",
                                    ifelse(summary_stats$p_value < 0.05, "**",
                                           ifelse(summary_stats$p_value < 0.10, "*", "")))

print(summary_stats)

# Save summary
write.csv(summary_stats, "summary_results.csv", row.names = FALSE)