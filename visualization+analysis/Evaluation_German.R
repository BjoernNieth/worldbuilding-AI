library(glmmTMB)
library(car)
library(emmeans)
library(ggplot2)
library(DHARMa)
library(dplyr)
library(tidyr)
library(patchwork)
library(MASS)
library(quantreg)

# Load the dataframe I prepared in the classifier script
# I brought my dataset into the longitudinal format, where story x section is one row
d <- read.csv("stories_long_Ger.csv")

# Replace story_id with story_id x model such that we can model the random effect stories might have 
d$story_uid <- interaction(d$story_id, d$model, drop = TRUE)
# Make section factor data, such that the model does not assume a linear connection between sections
d$section   <- factor(d$section, levels = 1:10)
# We want to test two models 1. Human vs AI 2. All models (where human is a "model")
# Therefore we build the indicatior 0/1 for is_human
d$is_human  <- ifelse(d$model == "Human", 1, 0)
# We make "Human" our reference model against which we comparea
d$model     <- relevel(factor(d$model), ref = "Human")

# We define our analyzis as a function such that we can repeat it for each outcome
# Outcome here is our space 
analyze_outcome <- function(data, outcome) {
  # This is a fancy printing header to be better able to read the script later
  cat("\n=========================================\n")
  cat("Analyzing outcome:", outcome, "\n")
  cat("=========================================\n")
  
  # Here we define both our models. The 1. simple model and the 2. full model
  # The simple model models human vs. AI section wise while taking in the story ID as a random effect
  # For the random effect we take just an intercept and no slope
  f_simple <- as.formula(paste(outcome, "~ is_human * section + (1 | story_uid)"))
  # The full model basically models the same, but takes each model category as a value 
  f_full   <- as.formula(paste(outcome, "~ model    * section + (1 | story_uid)"))
  
  # We give more iterations when fitting the model as we have 50k oberservations
  ctrl <- glmmTMBControl(optCtrl = list(iter.max = 1e4))
  
  # Our data is between [0,1] including both 0 and 1 for our observation
  # In Robert Kubinec 2023 the ordbeta model is introduced which deals exactly with such data
  # This model is in glmmTMB so we just use it for both models
  m_simple <- glmmTMB(f_simple, data = data, family = ordbeta(), control = ctrl)
  print("Summary simple model")
  print(summary(m_simple))
  
  m_full   <- glmmTMB(f_full,   data = data, family = ordbeta(), control = ctrl)
  print("Summary full model")
  print(summary(m_full))
  
  # Print a warning if there was a problem fitting the model 
  if (!is.null(m_simple$fit$convergence) && m_simple$fit$convergence != 0)
    warning("Simple model for ", outcome, " did not converge cleanly.")
  if (!is.null(m_full$fit$convergence) && m_full$fit$convergence != 0)
    warning("Full model for ", outcome, " did not converge cleanly.")
  
  # Plots the mean + 95% confidence interval for the true distribution mean
  mean_ci_fn <- function(x) {
    m <- mean(x)
    s <- sd(x)
    n <- length(x)
    data.frame(y = m,
               ymin = m - 1.96 * s / sqrt(n),
               ymax = m + 1.96 * s / sqrt(n))
  }
  
  # Gives a line plot for our observation for human vs ai
  p_desc_simple <- ggplot(data, aes(x = as.numeric(as.character(section)),
                                    y = .data[[outcome]],
                                    color = factor(is_human),
                                    fill  = factor(is_human))) +
    stat_summary(fun.data = mean_ci_fn, geom = "ribbon",
                 alpha = 0.2, color = NA) +
    stat_summary(fun = mean, geom = "line") +
    stat_summary(fun = mean, geom = "point") +
    scale_color_manual(values = c("red", "blue"), labels = c("LLM", "Human")) +
    scale_fill_manual( values = c("red", "blue"), labels = c("LLM", "Human")) +
    labs(title = paste("Descriptive:", outcome),
         x = "Narrative section",
         y = paste(outcome, "(mean ± 95% CI)"),
         color = "", fill = "") +
    theme_minimal()
  
  # Gives a line plot for our observation for all models (where human is a model)
  p_desc_full <- ggplot(data, aes(x = as.numeric(as.character(section)),
                                  y = .data[[outcome]],
                                  color = model, fill = model)) +
    stat_summary(fun.data = mean_ci_fn, geom = "ribbon",
                 alpha = 0.15, color = NA) +
    stat_summary(fun = mean, geom = "line") +
    stat_summary(fun = mean, geom = "point") +
    labs(title = paste("Descriptive:", outcome),
         x = "Narrative section",
         y = paste(outcome, "(mean ± 95% CI)"),
         color = "", fill = "") +
    theme_minimal()

  
  # Run Anova type 3 on the simpel model 
  # is_human: Do Humans differ from LLMs at section 1?
  # section: Is_human = 0 as first, do LLMs vary by section?
  # is_human:section: Does the gap between human and AI changes by section? 
  a_simple <- Anova(m_simple, type = 3)
  print("Simple Anova table")
  print(a_simple)
  
  # Run Anova type 3 on the full model
  # model: At section 1, do the models differ?
  # section: For Human, do the outcomes vary by section?
  # model:section: Does the section pattern differ across the 5 models?
  a_full   <- Anova(m_full,   type = 3)
  print("Full Anova table")
  print(a_full)
  
  
  # Compute Estimated Marginal Means (EMMeans) for the simple model
  # Basically get the model predictions for a specific combination of conditions
  # For simple model:
  # Fix section x, what is the value of response for human vs ai
  # Our GLM is on model scale, so we need to calculate back to % of space in narrative section
  # logit(mu_ij) = intercept + beta_model_i + beta_section_j + beta_(modelxsection)ij
  # SE --> Standard Error of experiment. Because 50k observation student-t -> normal
  # Lower CL and Upper CL with estimate +- 1.96*SE which is the 95% CI for a group mean under a gaussian distribution 
  em_s <- emmeans(m_simple, ~ is_human | section, regrid = "response")
  # Contrast is the difference between the means while taking in the influence of SE when taking a substract
  # z_ratio = estimate/SE. This gives how many SE estimate away from zero. 
  # Zero here is the H_0 because if the estimates are the same we would see no difference
  # R gives us the z.ratio, as z~N(0,1) because again the t-distribution collapse to the standard normal
  c_s  <- contrast(em_s, method = "revpairwise", adjust = "holm")
  print("Simple model emmeans")
  print(em_s)
  print("Simple model contrast")
  print(c_s)
  
  # Compute the same for the full model
  em_f         <- emmeans(m_full, ~ model | section, regrid = "response")
  # Compute the control vs test group which is ai-models vs human reference
  c_f_vs_human <- contrast(em_f, method = "trt.vs.ctrl", ref = "Human", adjust = "holm")
  # Compute all contrasts. Here we need to correct for the error through tukey as we have all possible combinations
  c_f_pairs    <- contrast(em_f, method = "pairwise",                  adjust = "tukey")
  print("Full model emmeans")
  print(em_f)
  print("Full model contrast test vs control")
  print(c_f_vs_human)
  print("Full model contrast pairwise")
  print(c_f_pairs)
  
  # Plot the model estimated mean with the assumed normal CI
  # This means, that we expect that the true group means lies with 95% within this interval
  ef_df <- as.data.frame(em_f)
  res <-ggplot(ef_df, aes(x = section, y = response, color = model, group = model)) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2) +
    geom_errorbar(aes(ymin = asymp.LCL, ymax = asymp.UCL),
                  width = 0.15, linewidth = 0.5) +
    labs(x = "Narrative section",
         y = paste("Proportion of", outcome),
         color = "Author") +
    theme_minimal(base_size = 12)
  plot(res)
  
  # --- Data frames for plotting ---
  em_df_s <- as.data.frame(emmeans(m_simple, ~ is_human * section, regrid = "response"))
  em_df_f <- as.data.frame(emmeans(m_full,   ~ model    * section, regrid = "response"))
  
  # --- Plots ---
  p_simple <- ggplot(em_df_s,
                     aes(x = as.numeric(as.character(section)), y = response,
                         color = factor(is_human), fill = factor(is_human))) +
    geom_ribbon(aes(ymin = asymp.LCL, ymax = asymp.UCL),
                alpha = 0.2, color = NA) +
    geom_line() + geom_point() +
    scale_color_manual(values = c("red", "blue"), labels = c("LLM", "Human")) +
    scale_fill_manual( values = c("red", "blue"), labels = c("LLM", "Human")) +
    labs(title = outcome, x = "Narrative section",
         y = "Estimated frequency", color = "", fill = "") +
    theme_minimal()
  plot(p_simple)
  
  p_full <- ggplot(em_df_f,
                   aes(x = as.numeric(as.character(section)), y = response,
                       color = model, fill = model)) +
    geom_ribbon(aes(ymin = asymp.LCL, ymax = asymp.UCL),
                alpha = 0.15, color = NA) +
    geom_line() + geom_point() +
    labs(title = outcome, x = "Narrative section",
         y = "Estimated frequency", color = "", fill = "") +
    theme_minimal()
  plot(p_full)
  
  list(
    outcome                  = outcome,
    model_simple             = m_simple,
    model_full               = m_full,
    anova_simple             = a_simple,
    anova_full               = a_full,
    contrasts_simple         = c_s,
    contrasts_full_vs_human  = c_f_vs_human,
    contrasts_full_pairwise  = c_f_pairs,
    emmeans_simple_df        = em_df_s,
    emmeans_full_df          = em_df_f,
    plot_simple              = p_simple,
    plot_full                = p_full
  )
}

# List of all spaces to test. We excluded no_space as this is 1 - all_space
outcomes <- c("perceived_space", "action_space", "visual_space",
              "descriptive_space", "all_space")
#outcomes <- c("perceived_space")

# Get the results for all outcomes
results <- lapply(outcomes, function(o) analyze_outcome(d, o))
names(results) <- outcomes

# Define different CSV to save the results in 
summary_simple <- do.call(rbind, lapply(results, function(r) {
  a <- r$anova_simple
  data.frame(
    outcome           = r$outcome,
    is_human_Chisq    = a["is_human",          "Chisq"],
    is_human_df       = a["is_human",          "Df"],
    is_human_p        = a["is_human",          "Pr(>Chisq)"],
    interaction_Chisq = a["is_human:section",  "Chisq"],
    interaction_df    = a["is_human:section",  "Df"],
    interaction_p     = a["is_human:section",  "Pr(>Chisq)"]
  )
}))
print(summary_simple, row.names = FALSE)
write.csv(summary_simple, "summary_omnibus.csv", row.names = FALSE)
all_contrasts_simple <- do.call(rbind, lapply(results, function(r) {
  df <- as.data.frame(r$contrasts_simple)
  df$outcome <- r$outcome
  df
}))
write.csv(all_contrasts_simple, "contrasts_human_vs_LLM.csv", row.names = FALSE)
all_contrasts_full_vs_human <- do.call(rbind, lapply(results, function(r) {
  df <- as.data.frame(r$contrasts_full_vs_human)
  df$outcome <- r$outcome
  df
}))
write.csv(all_contrasts_full_vs_human, "contrasts_each_LLM_vs_Human.csv", row.names = FALSE)
# Helper: convert an Anova table to a tidy data frame (terms become a column,
# not row names — easier to view, filter, and write to CSV)
anova_to_df <- function(a) {
  df <- as.data.frame(a)
  df$term <- rownames(df)
  rownames(df) <- NULL
  df[, c("term", setdiff(names(df), "term"))]
}

# For each outcome, create one data frame per printed table in the global env.
# Names follow the pattern <table>_<outcome>, e.g. anova_full_perceived_space
for (name in names(results)) {
  r <- results[[name]]
  
  # Anova tables
  assign(paste0("anova_simple_",            name), anova_to_df(r$anova_simple))
  assign(paste0("anova_full_",              name), anova_to_df(r$anova_full))
  
  # EMMs (already stored as data frames inside the result list)
  assign(paste0("emmeans_simple_",          name), r$emmeans_simple_df)
  assign(paste0("emmeans_full_",            name), r$emmeans_full_df)
  
  # Contrasts (stored as emmGrid objects — convert to data frame)
  assign(paste0("contrasts_simple_",        name), as.data.frame(r$contrasts_simple))
  assign(paste0("contrasts_full_vs_human_", name), as.data.frame(r$contrasts_full_vs_human))
  assign(paste0("contrasts_full_pairwise_", name), as.data.frame(r$contrasts_full_pairwise))
}
# Save every just-created data frame whose name starts with one of the table prefixes
prefixes <- c("anova_simple_", "anova_full_",
              "emmeans_simple_", "emmeans_full_",
              "contrasts_simple_", "contrasts_full_vs_human_", "contrasts_full_pairwise_")

dir.create("results_tables", showWarnings = FALSE)

for (nm in ls(pattern = paste0("^(", paste(prefixes, collapse = "|"), ")"))) {
  write.csv(get(nm), file.path("results_tables", paste0(nm, ".csv")),
            row.names = FALSE)
}
# Plot all AI vs human plots in a 5-grid
grid_simple <- wrap_plots(lapply(results, function(r) r$plot_simple),
                          ncol = 2, guides = "collect") &
  theme(legend.position = "bottom")
ggsave("grid_human_vs_LLM.png", grid_simple, width = 12, height = 12)

# Plot all model plots in a 5-grid
grid_full <- wrap_plots(lapply(results, function(r) r$plot_full),
                        ncol = 2, guides = "collect") &
  theme(legend.position = "bottom")
ggsave("grid_all_models.png", grid_full, width = 12, height = 12)

# --- Directory setup ---
dir.create("diagnostics",         showWarnings = FALSE)
dir.create("diagnostics/simple",  showWarnings = FALSE)
dir.create("diagnostics/full",    showWarnings = FALSE)

# --- Helper: run the full diagnostic battery on one fitted model ---
# Saves QQ plot, residual-vs-prediction plot, residual-by-group plot, and
# residual-by-section plot into out_dir. Returns a one-row summary data frame.
run_diagnostics <- function(fit, outcome, group_var, group_label, out_dir) {
  sim <- simulateResiduals(fit)
  
  png(file.path(out_dir, paste0(outcome, "_qq.png")),
      width = 800, height = 700)
  plotQQunif(sim, main = paste("QQ:", outcome))
  dev.off()
  
  png(file.path(out_dir, paste0(outcome, "_resid_pred.png")),
      width = 800, height = 700)
  plotResiduals(sim, quantreg = TRUE)
  dev.off()
  
  png(file.path(out_dir, paste0(outcome, "_resid_by_", group_label, ".png")),
      width = 1000, height = 700)
  plotResiduals(sim, form = group_var)
  dev.off()
  
  png(file.path(out_dir, paste0(outcome, "_resid_by_section.png")),
      width = 1000, height = 700)
  plotResiduals(sim, form = d$section)
  dev.off()
  
  # formal tests
  td <- testDispersion(sim,     plot = FALSE)
  tz <- testZeroInflation(sim,  plot = FALSE)
  to <- testOutliers(sim,       plot = FALSE)
  
  data.frame(
    outcome            = outcome,
    dispersion_stat    = unname(td$statistic),
    dispersion_p       = td$p.value,
    zeroinflation_stat = unname(tz$statistic),
    zeroinflation_p    = tz$p.value,
    outlier_p          = to$p.value
  )
}

# --- Run diagnostics for both the simple and the full model ---
# Collect per-outcome summary rows in two lists, one per model type.
diag_simple <- list()
diag_full   <- list()

# Build a labeled factor for the simple-model grouping so the boxplot
# reads "Human" / "LLM" rather than 1 / 0.
is_human_factor <- factor(d$is_human, levels = c(0, 1),
                          labels = c("LLM", "Human"))

for (name in names(results)) {
  cat("\n=== Diagnostics for:", name, "===\n")
  r <- results[[name]]
  
  cat("  [simple model: is_human * section]\n")
  diag_simple[[name]] <- run_diagnostics(
    fit         = r$model_simple,
    outcome     = name,
    group_var   = is_human_factor,
    group_label = "is_human",
    out_dir     = "diagnostics/simple"
  )
  
  cat("  [full model: model * section]\n")
  diag_full[[name]] <- run_diagnostics(
    fit         = r$model_full,
    outcome     = name,
    group_var   = d$model,
    group_label = "model",
    out_dir     = "diagnostics/full"
  )
}

# --- Summary tables ---
diag_table_simple            <- do.call(rbind, diag_simple)
diag_table_simple$model_type <- "simple"

diag_table_full              <- do.call(rbind, diag_full)
diag_table_full$model_type   <- "full"

# Combined table with a model_type column as the first column
diag_table_combined <- rbind(diag_table_simple, diag_table_full)
diag_table_combined <- diag_table_combined[
  , c("model_type", setdiff(names(diag_table_combined), "model_type"))
]

write.csv(diag_table_simple,
          "diagnostics/simple/diagnostic_summary_simple.csv",
          row.names = FALSE)

write.csv(diag_table_full,
          "diagnostics/full/diagnostic_summary_full.csv",
          row.names = FALSE)

write.csv(diag_table_combined,
          "diagnostics/diagnostic_summary_combined.csv",
          row.names = FALSE)

print(diag_table_combined)