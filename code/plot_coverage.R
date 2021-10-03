# =============================================================================
# Create Tables and Plots for CI Experiments
# =============================================================================
devtools::load_all()

# Where to put the plots
target_dir = sprintf("%s/paper/figures", here())

rename_problems = function(x){
  # otherwise can give problems with factors
  x = as.character(x)
  c("x12" = "linear", "x1234" = "non-linear")[x]
}

rename_algorithms = function(x){
  x = as.character(x)
  c("rpart" = "tree", "randomForest" = "rf", "lm" = "lm")[x]
}

rename_sampling = function(x){
  x = as.character(x)
  c("bootstrap" = "boot", "subsampling" = "subs", "ideal" = "ideal")[x]
}

gen_plot = function(dat, ndat, type, method){
  dat = dat[dat$n == ndat, ]
	if(type == "coverage"){
    txt = "Coverage"
		ymax = 1
	} else {
		txt = "Width"
		ymax = max(dat$avg_width) 
	}
  p = ggplot(dat,
         aes_string(x = "nrefits", y = type, color = "sampling_strategy",
                    shape = "adjusted", linetype = "adjusted")) + 
    facet_grid(problem ~ algorithm) + 
    geom_line() + 
    scale_y_continuous(sprintf("Confidence Interval %s", txt), limits = c(0, ymax)) +
    scale_x_continuous("Number of Model Refits") +
    scale_color_discrete("Sampling Strategy") 
	if(type == "coverage") p = p + geom_hline(yintercept = .95, linetype = "dashed")
  p
}


# Plotting function for coverage and width plot (Appendix)
plot_cov_width = function(dat, ndat, type, method){
  p = gen_plot(dat, ndat, type, method)
  plot(p)
  ggsave(sprintf("%s/ci-%s-%s-n%i.pdf", fig_dir, type,  method, ndat), width = 10, height = 6)
}


# =============================================================================
# Create results table for PFI
# =============================================================================
coverage_pfi_mean = readRDS(sprintf("%s/coverage_pfi_mean.Rds", res_dir))
coverage_pfi_mean$sampling_strategy = rename_sampling(coverage_pfi_mean$sampling_strategy)
coverage_pfi_mean$problem = rename_problems(coverage_pfi_mean$problem)
coverage_pfi_mean$algorithm = rename_algorithms(coverage_pfi_mean$algorithm)

res_pfi = filter(coverage_pfi_mean, nrefits == 15)
adjusted = ifelse(res_pfi$adjusted & (res_pfi$sampling_strategy != "ideal"), "*", "")
res_pfi$sampling_strategy = sprintf("%s%s", res_pfi$sampling_strategy, adjusted)
res_pfi = select(res_pfi, dgp = problem, model = algorithm, n, sampling_strategy, coverage)
res_pfi = res_pfi %>% tidyr::spread(sampling_strategy, coverage) %>%
		relocate(ideal, .after = last_col())
lab = "sim-ci-coverage-pfi"
cap = "Coverage Probability of the 95\\% PFI Confidence Intervals. boot = bootstrap, subs = subsampling, * = with adjustment."
filename = sprintf("%s/paper/figures/sim-ci-coverage-pfi.tex", here())
res_pfi %>%
  kbl(booktabs = TRUE, format = "latex", digits = 2,
      caption = cap, label = lab, escape = FALSE, linesep = "") %>%
  kable_classic() %>%
  write(file = filename)

# =============================================================================
# Coverage Plots for PFI
# =============================================================================
plot_cov_width(coverage_pfi_mean, 100, "coverage", "pfi")
plot_cov_width(coverage_pfi_mean, 1000, "coverage", "pfi")

# =============================================================================
# CI Width Plots for PFI
# =============================================================================
plot_cov_width(coverage_pfi_mean, 100, "avg_width", "pfi")
plot_cov_width(coverage_pfi_mean, 1000, "avg_width", "pfi")

# =============================================================================
# Create results table for PDP
# =============================================================================
coverage_pdp_mean = readRDS(sprintf("%s/coverage_pdp_mean.Rds", res_dir))
coverage_pdp_mean$sampling_strategy = rename_sampling(coverage_pdp_mean$sampling_strategy)
coverage_pdp_mean$problem = rename_problems(coverage_pdp_mean$problem)
coverage_pdp_mean$algorithm = rename_algorithms(coverage_pdp_mean$algorithm)

res_pdp = filter(coverage_pdp_mean, nrefits == 15)
adjusted = ifelse(res_pdp$adjusted & (res_pdp$sampling_strategy != "ideal"), "*", "")
res_pdp$sampling_strategy = sprintf("%s%s", res_pdp$sampling_strategy, adjusted)
res_pdp = select(res_pdp, dgp = problem, model = algorithm, n, sampling_strategy, coverage)
res_pdp = res_pdp %>% tidyr::spread(sampling_strategy, coverage) %>%
		relocate(ideal, .after = last_col())
lab = "sim-ci-coverage-pdp"
cap = "Coverage Probability of the 95\\% PDP Confidence Bands. boot = bootstrap, subs = subsampling, * = with adjustment."
filename = sprintf("%s/paper/figures/sim-ci-coverage-pdp.tex", here())
res_pdp %>%
  kbl(booktabs = TRUE, format = "latex", digits = 2,
      caption = cap, label = lab, escape = FALSE, linesep = "") %>%
  kable_classic() %>%
  write(file = filename)

# =============================================================================
# Create combined results table
# =============================================================================
x = merge(res_pdp, res_pfi, by = c("dgp", "model", "n"))
colnames(x) = gsub("\\.(x|y)", "", colnames(x))
lab = "sim-ci-coverage"
cap = "Coverage Probability of the 95\\% Confidence Bands/Intervals for PDP and PFI. boot = bootstrap, subs = subsampling, * = with adjustment."
filename = sprintf("%s/paper/figures/sim-ci-coverage.tex", here())
x %>%
  kbl(booktabs = TRUE, format = "latex", digits = 2,
      caption = cap, label = lab, escape = FALSE, linesep = "") %>%
	add_header_above(c("", "", "", "PD" = 5, "PFI" = 5)) %>%
  kable_classic() %>%
  write(file = filename)

# =============================================================================
# Coverage Plots for PDP
# =============================================================================
plot_cov_width(coverage_pdp_mean, 100, "coverage", "pdp")
plot_cov_width(coverage_pdp_mean, 1000, "coverage", "pdp")

# =============================================================================
# CI Width Plots for PDP
# =============================================================================
plot_cov_width(coverage_pdp_mean, 100, "avg_width", "pdp")
plot_cov_width(coverage_pdp_mean, 1000, "avg_width", "pdp")



# =============================================================================
# CI Width and Coverage for one scenario
# =============================================================================

x = filter(coverage_pdp_mean, algorithm == "rf", problem == "non-linear")
p1 = gen_plot(x, 1000, "avg_width", "pdp") + theme(legend.position = "none")
p2 = gen_plot(x, 1000, "coverage", "pdp")
p1 + p2
ggsave(sprintf("%s/ci-example.pdf", fig_dir), width = 10, height = 4)


# =============================================================================
# Coverage vs. Width
# =============================================================================

cpdp = coverage_pdp_mean
cpdp$type = "PD"
cpfi = coverage_pfi_mean
cpfi$type = "PFI"
x = rbind(cpdp, cpfi)

x$sampling_strategy = as.character(x$sampling_strategy)

x0 = x %>% filter(!(sampling_strategy == "ideal"),
                  nrefits == 15, !adjusted)
x = x %>%
  filter(!(sampling_strategy == "ideal"),
         nrefits == 15,
         adjusted)
x1 = x[sampling_strategy == "subs",] 
x2 = x[sampling_strategy == "boot",] 
xx = data.frame(xfrom = x1$avg_width,
                xto = x2$avg_width,
                yfrom = x1$coverage,
                yto = x2$coverage,
                l = sprintf("%s,%s,%i", x1$problem, x1$algorithm, x1$n),
                type = x1$type)
ggplot(x) +
  geom_segment(aes(x = xfrom, xend = xto, y = yfrom, yend = yto), alpha = 0.7, data = xx) +
#  geom_label(aes(x = xto, y = yto, label = l)size = 2, check_overlap = TRUE,  data = xx) +
  geom_point(aes(x = avg_width, y = coverage, color = sampling_strategy)) +
  geom_hline(yintercept = 0.95, lty = 2) +
  facet_wrap("type", scales = "free_x") +
  scale_x_continuous("CI width") + 
  scale_y_continuous("CI coverage", limits = c(0.2,NA)) +
  scale_color_discrete("Resampling")

ggsave(sprintf("%s/ci-width-vs-coverage.pdf", fig_dir), width = 8, height = 3)


cpdp = coverage_pdp_mean
cpdp$type = "PD"
cpfi = coverage_pfi_mean
cpfi$type = "PFI"
x = rbind(cpdp, cpfi)

x$sampling_strategy = as.character(x$sampling_strategy)

x = x %>%
  filter(!(sampling_strategy == "ideal"),
         nrefits == 15)
x1 = x[x$adjusted,] 
x2 = x[!(x$adjusted),] 
xx = data.frame(xfrom = x1$avg_width,
                xto = x2$avg_width,
                yfrom = x1$coverage,
                yto = x2$coverage,
                sampling_strategy= x1$sampling_strategy, 
                l = sprintf("%s,%s,%i", x1$problem, x1$algorithm, x1$n),
                type = x1$type)
ggplot(x) +
  geom_segment(aes(x = xfrom, xend = xto, y = yfrom, yend = yto), alpha = 0.7, data = xx) +
#  geom_label(aes(x = xto, y = yto, label = l)size = 2, check_overlap = TRUE,  data = xx) +
  geom_point(aes(x = avg_width, y = coverage, shape = adjusted, color = sampling_strategy), size = 2) +
  geom_hline(yintercept = 0.95, lty = 2) +
  facet_grid(. ~ type, scales = "free_x") +
  scale_x_continuous("CI width") + 
  scale_y_continuous("CI coverage") +
  scale_color_discrete("Resampling") +
  scale_shape_discrete("Variance corrected?")

ggsave(sprintf("%s/ci-width-vs-coverage-adjusted.pdf", fig_dir), width = 8, height = 3)
