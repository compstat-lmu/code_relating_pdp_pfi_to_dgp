# =============================================================================
# Visualize the Shrinkage bias
# =============================================================================

devtools::load_all()
# Visualize sources of biases and variance
n = 10000
dat = data.frame(x = seq(from = 0, to = 10, length.out = n))
dat$y = sin(dat$x) + 0.5 * dat$x  
# center y for simpler bias type computation
dat$y = dat$y - mean(dat$y)
dat$yplus = ifelse(dat$y > 0, 3, -3)
dat$wrong = -dat$yplus
dat$model = -1.1 * (dat$x < 5.7) + 1.7 * (dat$x >= 5.7)
dat_ribbon = data.frame(x = c(dat$x, dat$x, dat$x),
                        ymin = c(dat$y, dat$yplus, dat$wrong),
                        ymax = c(rep(0, n), dat$y, rep(0, n)),
                        type = rep(c("shrinkage", "", ""), each = n))
p = ggplot(dat) +
  geom_ribbon(aes(x = x, ymin = ymin, ymax = ymax, fill = type, group = type), data = dat_ribbon, alpha = 0.5) +
  geom_line(aes(x = x, y = y), color = "black",  size = 5) +
  geom_hline(aes(yintercept = 0), size = 3, lty = 2) +
  geom_line(aes(x = x, y = model), color = "black", size = 2.5) +
  scale_x_continuous("Feature value") +
  scale_y_continuous("Partial Dependence of Feature") + 
  scale_fill_manual("Bias", values = c("white","deepskyblue"),
                    labels = c("", "Shrinkage"), guide = FALSE) +
  # Arrow for PDP
  annotate("segment", x = 1, xend = 0.1, y = -2.5, yend = -2.5, colour = "blue", size=3, arrow=arrow()) +
  annotate("label", x = 1.4, y = -2.5, label = "True PDP", size = 12) +
  annotate("segment", x = 8, xend = 7, y = 0.3, yend = 0, colour = "blue", size=3, arrow=arrow()) +
  annotate("label", x = 8.5, y = 0.35, label = "Mean prediction", size = 12) +
  annotate("segment", x = 5, xend = 6.5, y = 2.35, yend = 1.7, colour = "blue", size=3, arrow=arrow()) +
  annotate("label", x = 5, y = 2.35, label = "Model PDP", size = 12) +
  theme_minimal() +
  theme(text = element_text(size = 30),
        axis.text.x= element_blank(),
        axis.text.y= element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())
filename = sprintf("%s/paper/figures/bias-types.jpeg", here())
jpeg(file = filename, width = 1200, height = 1000)
print(p)
dev.off()
