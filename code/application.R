# =============================================================================
# Application that is used throughout the paper
# Based on Bicycle count data for Munich
# Model: count ~ weather and season
# =============================================================================
# Optional TODO:
# - Add some pre-processing step, just for illustration (e.g. feature selection)
# - Season feature
# - weekday feature
# - Use bootstrapping instead of CV


devtools::load_all()
set.seed(42)

N_FOLDS = 15
# Confidence level for all CIs
alpha = 0.05
t_alpha = qt(1 - alpha / 2, df = N_FOLDS - 1)
#z_alpha = qnorm(1 - alpha / 2)


# =============================================================================
# Read Data
# =============================================================================
wine_dat = sprintf("%s/winequality-red.csv", data_dir)
dat = read.csv(wine_dat, sep = ";")
print(nrow(dat))

# Shuffle to remove the date order
dat = dat[sample(1:nrow(dat)),]

# Shorten some colnames
dat = rename(dat, 
             total.SO2 = total.sulfur.dioxide,
             free.SO2 = free.sulfur.dioxide,
             res.sugar = residual.sugar,
             vol.acid = volatile.acidity,
             fixed.acid = fixed.acidity)

# =============================================================================
# Train Models
# =============================================================================
lrn_lm = lrn("regr.lm")
lrn_ranger = lrn("regr.ranger", num.trees = 100)
lrn_tree = lrn("regr.rpart")
task = TaskRegr$new("wine", backend = dat, target = "quality")
# Later for PFI
# Using 0.623 to be comparable to bootstrapped models
train_index = sample(1:nrow(dat), size = 0.632 * nrow(dat), replace = FALSE)
test_index = setdiff(1:nrow(dat), train_index)

# =============================================================================
# Benchmark Models
# =============================================================================
learners = list(lrn_lm, lrn_tree, lrn_ranger)
n1 = 0.632 *  task$nrow
n2 = (1 - 0.632) * task$nrow
resamplings = rsmp("bootstrap", repeats = N_FOLDS, ratio = 1)
#resamplings = rsmp("subsampling", repeats = N_FOLDS, ratio = 0.632)
design = benchmark_grid(list(task), learners, resamplings)
bmr = benchmark(design, store_models = TRUE)



# Compute CIs and result table
mses = bmr$score() %>%
  dplyr::group_by(learner_id, task) %>%
  dplyr::summarize(mse = mean(regr.mse))

model_names = c("regr.lm" = "Linear regression",
                "regr.ranger" = "Random Forest",
                "regr.rpart" = "Tree")
mses$learner_id = model_names[mses$learner_id]

text_mses = paste0(sprintf("%.3f (%s)", mses[["mse"]], mses[["learner_id"]]), collapse = ", ")
print(text_mses)
write(text_mses, file = sprintf("%s/results/application-model-comparison.tex", here()))

# Confidence interval of differences
scores = bmr$score()
scores_ranger = scores[learner_id == "regr.ranger",]
scores_lm = scores[learner_id == "regr.lm",]
scores_rpart = scores[learner_id == "regr.rpart",]
sdiff1 = scores_ranger$regr.mse - scores_lm$regr.mse
lower1 = mean(sdiff1) - t_alpha * sqrt((1/N_FOLDS + n2/n1) * var(sdiff1))
upper1 = mean(sdiff1) + t_alpha * sqrt((1/N_FOLDS + n2/n1) * var(sdiff1))

sdiff2 = scores_ranger$regr.mse - scores_rpart$regr.mse
lower2 = mean(sdiff2) - t_alpha * sqrt((1/N_FOLDS + n2/n1) * var(sdiff2))
upper2 = mean(sdiff2) + t_alpha * sqrt((1/N_FOLDS + n2/n1) * var(sdiff2))

text1 = sprintf("[%.3f;%.3f]", lower1, upper1)
text2 = sprintf("[%.3f;%.3f]", lower2, upper2)

write(text1, file = sprintf("%s/application-mse-diff-rf-lm.tex", res_dir))
write(text2, file = sprintf("%s/application-mse-diff-rf-tree.tex", res_dir))


# =============================================================================
# Model PDP
# =============================================================================
feature = "alcohol"

# Selecting the winner from benchmark
learner = lrn_ranger
# Retraining on entire task
learner$train(task, row_ids = train_index)

pred = Predictor$new(learner, data = dat[test_index,], y = "quality")
#eff = FeatureEffects$new(pred, method = "pdp")

pdp_dat = pdp_ci(pred, feature)
t_alpha_n = qt(1 - alpha / 2, df = length(test_index - 1))
pdp_dat$lower = pdp_dat$pdp - t_alpha_n * sqrt(pdp_dat$pdp_est_var)
pdp_dat$upper = pdp_dat$pdp + t_alpha_n * sqrt(pdp_dat$pdp_est_var)


# =============================================================================
# Compute var and confidence interval for algo PDP
# =============================================================================
bmr$filter(learner_ids = "regr.ranger")
rs = bmr$resample_result(1)
pdp_dat2 = RsmpPdp$new(rs, feature)

pdp_dat$type = "model-PDP"
pdp_dat2$variances$type = "learner-PDP"
pdps = rbindlist(list(pdp_dat, pdp_dat2$variances), fill = TRUE)


p_pdp = ggplot(pdps, aes_string(x = feature)) +
  geom_line(aes(y = pdp)) +
  geom_line(aes(y = upper), lty = 2) + 
  geom_line(aes(y = lower), lty = 2) +
  scale_y_continuous("PD") +
  facet_wrap("type")


#pdf(file = sprintf("%s/paper/figures/application-model-plus-algo-pdp.pdf", here()), width = 6, height = 3)
#plot(p_pdp)
#dev.off()



# =============================================================================
# Compute var and CI for model-PFI
# =============================================================================
vars_high = pfi_var(learner, task, nperm = 5, row_ids = test_index)

# order level by pfi
lv_order = order(vars_high$pfi)
vars_high$feature = factor(vars_high$feature, levels =  vars_high$feature[lv_order])

p = ggplot(vars_high, aes(y = feature)) + 
  geom_point(aes(x = pfi)) +
  geom_segment(aes(yend = feature, x = lower, xend = upper)) +
  scale_x_continuous("Permutation Feature Importance (MSE)") +
  scale_y_discrete("")

# =============================================================================
# Compute var and CI for algo-PFI
# =============================================================================
algo_pfi = rsmp_imp(rs, nperm = 5)
algo_pfi$feature = factor(algo_pfi$feature, levels = algo_pfi$feature[lv_order])

vars_high$type = "model-PFI"
algo_pfi$type = "learner-PFI"
xx = rbindlist(list(vars_high, algo_pfi), fill = TRUE)

p_pfi = ggplot(xx, aes(y = feature)) +
  geom_point(aes(x = pfi)) +
  geom_segment(aes(yend = feature, x = lower, xend = upper)) +
  scale_x_continuous("Permutation Feature Importance (MSE)") +
  scale_y_discrete("") +
  facet_wrap("type")


#pdf(file = sprintf("%s/paper/figures/application-model-plus-algo-pfi.pdf", here()), width = 6, height = 3)
#plot(p_pfi)
#dev.off()

p_pfi / p_pdp
ggsave(sprintf("%s/application.pdf", fig_dir), width = 8, height = 6)

