
# reproduce the data from Zhao
zhao <- tribble(
  ~query, ~true_prob, ~exp1, ~exp2, ~sim,
  "p_B", .3, .31, .29, .32,
  "p_B", .6, .58, .59, .59,
  "p_B", .9, .86, .85, .86,
  "p_AandB", .1, .19, .27, .15,
  "p_AandB", .4, .50, .51, .42,
  "p_AandB", .8, .80, .78, .77,
  "p_AgB", .33, .35, .33, .36,
  "p_AgB", .67, .63, .63, .65,
  "p_AgB", .89, .82, .83, .84,
  "ratio", .33, .70, .65, .44,
  "ratio", .67, .93, 1.00, .71,
  "ratio", .89, 1.03, 1.15, .90
) %>% 
  mutate(
    conj = ifelse(query=="p_AandB", 1,0),
    avg = (exp1+exp2)/2
    )

calc_ptn <- function(p, d, delta_d=0){
  d = d + delta_d
  
  (1-2*d)*p + d
}

ptn_loss <- function(data,par){
  with(
    data,
    {
    d_par <- par[1] + conj*par[2]
    preds <- calc_ptn(true_prob, d_par)
    return(sum( (avg-preds)^2 ))
    }
  )
}

res <- optim(c(.001,.001), ptn_loss %>% filter(query!="ratio"), data=zhao, method="SANN") # this was being grumpy

zhao <- zhao %>% 
  mutate(preds = calc_ptn(true_prob, res$par[1] + conj*res$par[2]))

zhao$preds[10:12] <- c(.553, .716, .872) # manually insert ratio predictions calc by hand

## compare results of their model vs mine
zhao %>% 
  select(exp1, exp2, true_prob, sim, preds) %>% 
  cor()

## not enough data points to compare statistically, but seems my version is 
## at least as good as theirs.

## ---------
##  just showing the math works for the inequalities
d = .2
## inequality 9
calc_ptn(.5, d) + calc_ptn(.25, d, .0) - calc_ptn(.75, d, .0)

## inequality 13
calc_ptn(.3, d, 0) - calc_ptn(.6, d)*calc_ptn(.5,d)

         