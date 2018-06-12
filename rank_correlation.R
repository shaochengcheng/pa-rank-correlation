require(igraph)

rank_correlation <- function(n, md){
  N <- 2 ^ n
  g <- barabasi.game(N, m=md, out.pref=TRUE, directed=FALSE) 
  dgr <- degree(g, v=V(g), mode='all', normalized=TRUE)
  btn <- betweenness(g, v = V(g), directed = FALSE, weights = NULL, nobigint = TRUE, normalized = TRUE)
  return(c(cor(dgr, btn, method='pearson'), cor(dgr, btn, method='spearman'), cor(dgr, btn, method='kendall')))
}

df <- data.frame(row.names=c('db_p', 'db_s', 'db_k'))
for (i in seq(1, 20)) {
  rbind(df, rank_correlation(10, 5))
}
