# Auxiliary functions
get_current_file_path <- function() {
  this_file = grep("^--file=", commandArgs(), value = TRUE)
  this_file = gsub("^--file=", "", this_file)
  if (length(this_file) == 0)
    this_file = rstudioapi::getSourceEditorContext()$path
  return(dirname(this_file))
}

maybe_install <- function(pkg) {
  if (length(find.package(pkg, quiet = TRUE)) == 0) {
    install.packages(pkg, repos = "http://cran.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

maybe_install("AER")
maybe_install("ggplot2")

# In a real scenario the dataset should be stripped of most likely duplicates
# https://lorentzen.ch/index.php/2021/04/16/a-curious-fact-on-the-diamonds-dataset/

# LM
diamonds.lm <- lm(
  price ~ carat + clarity + color,
  data = diamonds
)
# summary(diamonds.lm)

# GLM
data("CreditCard")
cc.glm <- glm(
  card ~ income + selfemp,
  # card ~ reports + dependents + active,
  data = CreditCard, family = binomial()
)
# summary(cc.glm)

# Export
export_to_compare <- function(model) {
  mf <- model.frame(model$terms, model$model)
  mm <- model.matrix(model$terms, mf)
  my <- model.response(mf)
  if (is.factor(my))
    my <- as.numeric(my) - 1

  data.table::fwrite(
    cbind(mm, my),
    file = paste0(get_current_file_path(), "/", class(model)[1], "_mm.csv")
  )

  data.table::fwrite(
    as.data.frame(
      model$coefficients,
      row.names = as.character(1:length(model$coefficients))
    ),
    file = paste0(get_current_file_path(), "/", class(model)[1], "_beta.csv")
  )
}

export_to_compare(diamonds.lm)
export_to_compare(cc.glm)
