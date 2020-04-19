#MidTerm 1952: Ranking the importance of Convolutional Neural Network Filters
#Chris Hamblin

# See the midterm R markdown document for a thorough explaination of the what and why of this code


#data loading
if (!require("feather")) {install.packages("feather"); require("feather")}      # package for sharing data between pandas and R
ranks_df <- read.csv('cifar_prunned_ranks.csv')   

numbers <- c('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20')
letters <- c('a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z')

#ranks_df$class <- factor(ranks_df$class, levels = c(numbers,letters))
ranks_df$class <- factor(ranks_df$class)
ranks_df$filter_num <- factor(ranks_df$filter_num)
ranks_df$filter_num_by_layer <- factor(ranks_df$filter_num_by_layer)
ranks_df$layer <- factor(ranks_df$layer)


#A wide version of our dataframe, where each class label gets a different column for its rank activations
if (!require("dplyr")) {install.packages("dplyr"); require("dplyr")}
if (!require("tidyr")) {install.packages("tidyr"); require("tidyr")}

ranks_df_wide <- ranks_df %>% select(filter_num, layer, filter_num_by_layer, prune_score, class) %>% 
  pivot_wider(names_from = class, values_from = prune_score)

#column names being numbers will be annoying later on, lets change to word version of numbers
num2word <- list('one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen',
                 'seventeen','eighteen','ninteen','twenty')
for (i in 1:20) {
  names(ranks_df_wide)[names(ranks_df_wide) == as.character(i)] <- num2word[i]
}

#getting column names in a sensible order
ranks_df_wide <- ranks_df_wide[c('filter_num','layer','filter_num_by_layer','one','two','three','four','five','six',
                                 'seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen',
                                 'seventeen','eighteen','ninteen','twenty','a','b','c','d','e','f','g','h','i','j','k','l','m',
                                 'n','o','p','q','r','s','t','u','v','w','x','y','z','overall')]


#plot of all filter ranks, colorcoded by image class. This plot is a bit crazy, but we just want a look at all the data. The x axis doesnt
#encode anything metric as each point is a unique filter. Vertical Lines indicate the boundaries of different layers
if (!require("ggplot2")) {install.packages("ggplot2"); require("ggplot2")}


plot_dataset_activations = function(data,title) {
  return(
    ggplot(data, aes(filter_num,prune_score)) + 
      geom_point(aes(colour = class, alpha=.5)) + ggtitle(title) + 
      geom_vline(aes(xintercept=45)) + geom_vline(aes(xintercept=68)) + 
      geom_vline(aes(xintercept=117))  +
      theme(axis.text.x=element_blank())
  )
}

plot_dataset_activations(ranks_df,'All Activation Map Ranks of Cifar CNN')

#What I immediately notice from this graph is the outlier filters, indicated by the long vertical streaks. The fact that many classes pop out in these
#streaks tell us that these filters are just very important in general, lots of different classes would be misclassified if these filters 
#were removed. Whats more these important neurons are much more prevalent in the early layers.

#Dont what to make too many of these grotesquely broad scatter plots, but lets just visualize two more where we separate out the letter classification
#ranks and the enumeration ranks

numbers_scat <- plot_dataset_activations(ranks_df[which(ranks_df$class %in% numbers),],'Enumeration Filter Ranks of CNN')
letters_scat <- plot_dataset_activations(ranks_df[which(ranks_df$class %in% letters),],'Letter Filter Ranks of CNN')
overall_scat <- plot_dataset_activations(ranks_df[which(ranks_df$class == 'overall'),],'Overall Filter Ranks of CNN')
if (!require("gridExtra")) {install.packages("gridExtra"); require("gridExtra")}
grid.arrange(numbers_scat,letters_scat, nrow = 1)

#Actually not that informative

#Histogram of all PIMs to get distribution

ggplot(ranks_df, aes(prune_score)) + geom_histogram() + ggtitle('Distribution of all Prune Scores')

#How do we know if our rank activation measure is capturing something meaningful? One way to do this is to look for correlations between certain
#classes. For example, we would expect the rank activation scores filters get for classifying 12 dots should be very similar for classifying 13 dots.
#After all, if a filter is important for classifying 12 dots we would expect it to be similarly important for classifying 13 dots. Similarly for classifying
#letters, we might expect 'b' and 'd' to have similar rank activation scores across filters, as they have a similar shape. However, we would not expect
#the ranks across filter to be correlated for say 'g' and '10' as presumably a very different sequence of filters is needed for counting dots and
#recognizing g's. So lets check out the correlations between a few select examples.


scat_11v12 <- ggplot(ranks_df_wide, aes(eleven, twelve)) + ggtitle('11 vs 12') +
  geom_point(aes(colour = layer)) +
  geom_smooth(method = "lm")


scat_gvq <- ggplot(ranks_df_wide, aes(g, q)) + ggtitle('g vs q') +
  geom_point(aes(colour = layer)) +
  geom_smooth(method = "lm")


scat_13vn <- ggplot(ranks_df_wide, aes(thirteen, n)) + ggtitle('13 vs n') +
  geom_point(aes(colour = layer)) +
  geom_smooth(method = "lm")

grid.arrange(scat_11v12, scat_gvq, scat_13vn, nrow = 1)


#These ggplots seem to show what we would expect, with a good fit between '11' and '12', 
#and a poor fit between '13' and 'n'. Lets fit the model more formally with the lme4 package. 
#We'll include a random slope and intercept effect for the layer each point belongs to,
#as we can imagine the relationship between class PIMs changes across layers.

#adding average response
ranks_df_wide$all_classes <- rowMeans(ranks_df_wide[4:49])



if (!require("lme4")) {install.packages("lme4"); require("lme4")}
if (!require("optimx")) {install.packages("optimx"); require("optimx")}

####fitting mixed effects models
set.seed(2)
fitgvq <- lmer(q ~ g + (g|layer), data = ranks_df_wide[which(ranks_df_wide$all_classes>.01),], REML=FALSE,
               control = lmerControl(
                 optimizer ='optimx', optCtrl=list(method='L-BFGS-B')))
summary(fitgvq) 
coef(fitgvq)

glmfitgvq <- glmer(q ~ g + (g|layer), data = ranks_df_wide[which(ranks_df_wide$all_classes>.01),],family=Gamma)


fitallvq <- lmer(q ~ all_classes + (all_classes|layer), data = ranks_df_wide[which(ranks_df_wide$all_classes>.01),], REML=FALSE)
summary(fitallvq) 
coef(fitallvq)

gvq_resid_plot <- plot(fitgvq)
allvq_resid_plot <- plot(fitallvq)
grid.arrange(gvq_resid_plot,allvq_resid_plot,nrow=1)

anova(fitgvq,fitallvq,ddf="Kenward-Roger") 

#Correlation Matrix Stuff

#full Correlation matrix of all classess
if (!require("corrplot")) {install.packages("corrplot"); require("corrplot")}
#full_correlation = cor(ranks_df_wide %>% select(4:50))
full_correlation = cor(ranks_df_wide %>% select(4:14))
corrplot(full_correlation, order = "hclust")

#Letters only
letters_correlation = cor(ranks_df_wide %>% select(24:49))
corrplot(letters_correlation, order = "hclust")

#hierarchical clustering
full_dissimilarity = 1 - full_correlation
dissimilarity_distances = as.dist(full_dissimilarity)
plot(hclust(dissimilarity_distances), main="Hierarchical clustering of Cifar10 Prune Scores", xlab="")


#models as subjects
multi_df <- read_feather('letterandenum_multimodel_ranks.feather') 

multi_df$class <- factor(multi_df$class, levels = c(numbers,letters))
multi_df$filter_num <- factor(multi_df$filter_num)
multi_df$filter_num_by_layer <- factor(multi_df$filter_num_by_layer)
multi_df$layer <- factor(multi_df$layer)
multi_df$model <- factor(multi_df$model)

multi_df_wide <- multi_df %>% select(filter_num, layer, filter_num_by_layer, PIM, class, model) %>% 
  pivot_wider(names_from = class, values_from = PIM)

for (i in 1:20) {
  names(multi_df_wide)[names(multi_df_wide) == as.character(i)] <- num2word[i]
}
#getting column names in a sensible order
multi_df_wide <- multi_df_wide[c('filter_num','layer','filter_num_by_layer','model','one','two','three','four','five','six',
                                 'seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen',
                                 'seventeen','eighteen','ninteen','twenty','a','b','c','d','e','f','g','h','i','j','k','l','m',
                                 'n','o','p','q','r','s','t','u','v','w','x','y','z')]

ggplot(multi_df_wide, aes(a, e)) + ggtitle('PIMs of "a" versus "e"') +
  geom_point(aes(colour = model)) +
  geom_smooth(method = "lm")

ggplot(multi_df_wide, aes(seventeen, i)) + ggtitle('PIMs of "seventeen" versus "i"') +
  geom_point(aes(colour = model)) +
  geom_smooth(method = "lm")

if (!require("lme4")) {install.packages("lme4"); require("lme4")}
if (!require("nlme")) {install.packages("nlme"); require("nlme")}  
if (!require("plyr")) {install.packages("plyr"); require("plyr")}  



multi_model_lme <- lme(l ~ twelve, random= ~ 1 + twelve | model, data = multi_df_wide,method="REML")
summary(multi_model_lme) 
plot(multi_model_lme)

coef1 <- coef(multi_model_lme) 
coef1
with(multi_df_wide, plot(e ~ a, xaxt = "n", col = "gray",
                         xlab = "a", ylab = "e", main = "Individual Regression Lines"))

abline(a = fixef(multi_model_lme)[1], b = fixef(multi_model_lme)[2], lwd = 2, col = "salmon")    
for (i in 1:nrow(coef1)) abline(a = coef1[i,1], b = coef1[i,2], col = "cadetblue")   ## individual regression lines
