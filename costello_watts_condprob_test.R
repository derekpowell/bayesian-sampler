
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


p_AB <- .8
p_AnotB <- .1
p_notAB <- .05
p_notAnotB <- .05

p_A <- p_AB + p_AnotB
p_B <- p_AB + p_notAB
p_AgB <- p_AB/p_B
p_BgA <- p_AB/p_A

##  just showing the math works for the equalities
## if events are independent/equiprobable, then equalities all work with base
## PT+N model. And even when events are mildly dependent, they are still approx right
## p116, their prediction is about expectation, so could show this by simulation

d = .24

## identity 3 (= 0)
identity_03 <- function(probs, d){
  p_AB <- probs[,1]
  p_AnotB <- probs[,2]
  p_notAB <- probs[,3]
  p_notAnotB <- probs[,4]
  
  p_A <- p_AB + p_AnotB
  p_B <- p_AB + p_notAB
  p_AgB <- p_AB/p_B
  p_BgA <- p_AB/p_A
  
  calc_ptn(p_AgB,d)*calc_ptn(p_B, d) - calc_ptn(p_BgA, d)*calc_ptn(p_A,d)
}

## identity 8 (= 0)
identity_08 <- function(probs, d){
  p_AB <- probs[,1]
  p_AnotB <- probs[,2]
  p_notAB <- probs[,3]
  p_notAnotB <- probs[,4]
  
  p_A <- p_AB + p_AnotB
  p_B <- p_AB + p_notAB
  p_AgB <- p_AB/p_B
  p_BgA <- p_AB/p_A
  p_BgnotA <- p_notAB/(1-p_A)
  p_AgnotB <- p_AnotB/(1-p_B)
  
  # calc_ptn(p_AgB,d)*calc_ptn(p_B, d) - calc_ptn(p_BgA, d)*calc_ptn(p_A,d)
  calc_ptn(p_AgnotB, d) + calc_ptn(p_B, d) + calc_ptn(p_BgnotA,d)*calc_ptn(p_A,d) -
    (calc_ptn(p_BgnotA, d) + calc_ptn(p_A, d) + calc_ptn(p_AgnotB,d)*calc_ptn(p_B,d) )
}


## identityuality 9 (=d)
identity_09 <- function(probs, d){
  p_AB <- probs[,1]
  p_AnotB <- probs[,2]
  p_notAB <- probs[,3]
  p_notAnotB <- probs[,4]
  
  p_A <- p_AB + p_AnotB
  p_B <- p_AB + p_notAB
  p_AgB <- p_AB/p_B
  p_BgA <- p_AB/p_A
  
  calc_ptn(p_A, d) + calc_ptn(p_notAB, d, .0) - calc_ptn((1-p_notAnotB), d, .0)
}



## identityuality 13 (=d/2)
identity_13 <- function(probs, d){
  p_AB <- probs[,1]
  p_AnotB <- probs[,2]
  p_notAB <- probs[,3]
  p_notAnotB <- probs[,4]
  
  p_A <- p_AB + p_AnotB
  p_B <- p_AB + p_notAB
  p_AgB <- p_AB/p_B
  p_BgA <- p_AB/p_A
  
  calc_ptn(p_AB, d, 0) - calc_ptn(p_BgA, d)*calc_ptn(p_A,d)
}

## identityuality 14 (=d/2)
identity_14 <- function(probs, d){
  p_AB <- probs[,1]
  p_AnotB <- probs[,2]
  p_notAB <- probs[,3]
  p_notAnotB <- probs[,4]
  
  p_A <- p_AB + p_AnotB
  p_B <- p_AB + p_notAB
  p_AgB <- p_AB/p_B
  p_BgA <- p_AB/p_A
  
  calc_ptn(p_AB, d, 0) - calc_ptn(p_AgB, d)*calc_ptn(p_B,d)
}


## identityuality 15 (=d/2)

identity_15 <- function(probs, d){
  p_AB <- probs[,1]
  p_AnotB <- probs[,2]
  p_notAB <- probs[,3]
  p_notAnotB <- probs[,4]
  
  p_A <- p_AB + p_AnotB
  p_B <- p_AB + p_notAB
  p_AgB <- p_AB/p_B
  p_BgA <- p_AB/p_A
  
  calc_ptn(p_AB, d, 0) - calc_ptn(p_A,d) + calc_ptn(p_AnotB/(1-p_B), d) *calc_ptn((1-p_B),d)
}


## identityuality 16 (=d/2)
identity_16 <- function(probs, d){
  p_AB <- probs[,1]
  p_AnotB <- probs[,2]
  p_notAB <- probs[,3]
  p_notAnotB <- probs[,4]
  
  p_A <- p_AB + p_AnotB
  p_B <- p_AB + p_notAB
  p_AgB <- p_AB/p_B
  p_BgA <- p_AB/p_A
  
  calc_ptn(p_AB, d, 0) - calc_ptn(p_B,d) + calc_ptn(p_notAB/(1-p_A), d) *calc_ptn((1-p_A),d)
}

## checking things here for some example identities, and all key d/2 involving conditionals
## identities 13-16 are the key ones

d = .4 # or whatever you like
mean(identity_13(rdirichlet(1e5, rep(1,4)), d))
mean(identity_14(rdirichlet(1e5, rep(1,4)), d))
mean(identity_15(rdirichlet(1e5, rep(1,4)), d))
mean(identity_16(rdirichlet(1e5, rep(1,4)), d))
