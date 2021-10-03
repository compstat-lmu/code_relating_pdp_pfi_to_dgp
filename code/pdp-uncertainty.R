# =============================================================================
# Create Figure to show sources of uncertainty for PDP
# =============================================================================
devtools::load_all()

# Number of data points
n = 200

# Definition of the partial function
f1 = function(x1) sin(x1) + 0.5 * x1
# E[Y|X]
f = function(x1, x2, x3) f1(x1) +  x2 + x2*x3^2
eps = function(n) rnorm(n, sd = 5)

#' Simulate data
#' 
#' x1 is drawn uniformly, x2 from Normal distribution, y defined by f
#'
#' @param n number of data points to sample
#' @return data.frame with x1, x2 and y
sim_data = function(n) {
  x1 = runif(n, min = 0, max = 10)
  x2 = rnorm(n)
  x3 = rnorm(n)
  y = f(x1, x2, x3) + eps(n)
  data.frame(x1, x2, x3, y)
}

#' Simulate a linear model for a new sample of the data
#'
#' @param i some id
#' @param seed the seed to set
#' @param mod_only If TRUE, return the linear model, else a data.frame with PDP values
#' @return data.frame with PDP grid, values and the id
sim = function(i, seed = 1, mod_only = FALSE){
  set.seed(seed)
  dat = sim_data(n)
  learner = lrn("regr.svm")
  task = TaskRegr$new(id = "sim", dat, target = "y")
  learner$train(task)
  if(mod_only){
    learner
  } else {
    pred = Predictor$new(learner, data = dat)
    fj = FeatureEffect$new(pred, feature = "x1", method = "pdp")
    x = seq(from = 0, to = 10, length.out = 70)
    fout = fj$predict(x)
    data.frame(x1 = x, .value = fout, i = i)
  }
}

sims = lapply(1:10, function(i) sim(i, seed = i))
dat_fhs = data.table::rbindlist(sims)
# Drop where we are outside of estimated PDP
dat_fhs = na.omit(dat_fhs)
dat_fhs$type = "PDP fh"
mean_pdps = dat_fhs[,list(y = mean(.value)), by = "x1"]
mean_pdps$type = "E(fh PDP)"

# Groundtruth
x1 = runif(n, min = 0, max = 10)
x2 = rnorm(n)
x3 = rnorm(n)
y = f(x1, x2, x3) + eps(n)
dat = data.frame(x1 = x1, y = f1(x1), type = "ftrue")

minx = min(dat_fhs$.value, f1(x1), na.rm = TRUE)
maxx = max(dat_fhs$.value, f1(x1), na.rm = TRUE)
ylims = c(minx, maxx)

p1 = ggplot(dat) +
  geom_line(aes(x = x1, y = .value, group = i), color = "black", data = dat_fhs, alpha = 0.5) +
  geom_line(aes(x = x1, y = y), color = "blue", size = 2,  data = mean_pdps) +
  geom_line(aes(x = x1, y = y), color = "green3", size = 2, lty = 2) +
  # The errors for y
  #geom_point(aes(x = x1, y = y), data = data.frame(x1,y)) +
  scale_y_continuous("PD", limits = ylims) +
  scale_color_discrete(guide = "none") +
  ggtitle("Bias and variance")

# Now we show one specific plot and how much the PDP estimate varies
i = 10
lmi = sim(i = i, seed = i, mod_only = TRUE)

pdpsi = lapply(1:5, function(i){
   dat = sim_data(50)
   pred = Predictor$new(lmi, dat)
   eff = FeatureEffect$new(pred, method = "pdp", feature = "x1")
   res = eff$results
   res$i = i
   res
})

pdpsi = data.table::rbindlist(pdpsi)

p2 = ggplot(pdpsi) +
  geom_line(aes(x = x1, y = .value, group = i)) +
  scale_y_continuous("PD", limits = ylims) +
  ggtitle("Uncertainty due to Monte Carlo integration")


## ICE plots for the same lm
dat = sim_data(200)
pred = Predictor$new(lmi, dat)
eff = FeatureEffect$new(pred, feature = "x1", method = "pdp+ice")
p3 = eff$plot() +
  scale_y_continuous(limits = ylims) +
  ggtitle("ICE curves")

pdf(file = sprintf("%s/paper/figures/pdp-uncertainty-sources.pdf", here()), width = 10, height = 4)
print(p2 + p1)
dev.off()

png(file = sprintf("%s/paper/figures/pdp-uncertainty-sources.png", here()), width = 2100, height = 700)
print(p2 + p1)
dev.off()




# =============================================================================
# Confidence intervals (point-wise)
# =============================================================================
sims = lapply(1:30, function(i) sim(i, seed = i))
dat_fhs = data.table::rbindlist(sims)
# Drop where we are outside of estimated PDP
dat_fhs = na.omit(dat_fhs)

dat2 = dat_fhs[,list(q025=quantile(.value, probs = 0.025),
                     q975=quantile(.value, probs = 0.975),
                     m=mean(.value)),by=x1]

p4 = ggplot(dat2, aes(x = x1)) +
  geom_line(aes(y = m)) +
  geom_line(aes(y=q025), lty = 2) +
  geom_line(aes(y=q975), lty = 2) +
  scale_y_continuous("y", limits = ylims)

# Now we show one specific plot and how much the PDP estimate varies
i = 10
lmi = sim(i = i, seed = i, mod_only = TRUE)
dat = sim_data(200)
pred = Predictor$new(lmi, dat)
eff = FeatureEffect$new(pred, method = "ice", feature = "x1")$results
z = function(x){sqrt(length(x)) * mean(x) / sd(x)}
eff = data.table::data.table(eff)
cis = eff[,list(z = mean(.value),
                q025 = t.test(.value)$conf.int[1],
                q975 = t.test(.value)$conf.int[2]),by = x1]

p5 = ggplot(cis, aes(x = x1)) +
  geom_line(aes(y = z)) +
  geom_line(aes(y=q025), lty = 2) +
  geom_line(aes(y=q975), lty = 2) +
  scale_y_continuous("", limits = ylims)
print(p5)

pdf(file = sprintf("%s/paper/figures/pdp-ci.pdf", here()), width=5, height=3)
print(p4 + p5)
dev.off()

png(file = sprintf("%s/paper/figures/pdp-ci.png", here()), width=2000, height=1000)
print(p4 + p5)
dev.off()

