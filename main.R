data = read.csv("dataset.csv")

set.seed(42)
data <- data[sample(nrow(data), 1500), ]

str(data)

data <- data[complete.cases(data),]
dim(data)

data = data[,c('sex','age','weight','height','HDL_chole','LDL_chole','triglyceride','hemoglobin','DRK_YN')]
str(data)

colnames(data)[colnames(data) == 'DRK_YN'] <- 'Y'


#Tolygus kintamieji padaromi i diskreciuosius
data$age_group <- ifelse(data$age >= 20 & data$age <= 35, '20-35',
                         ifelse(data$age > 35 & data$age <= 50, '35-50', '50+'))
data <- data[, !(colnames(data) %in% 'age')]

data$weight_group <- ifelse(data$weight >= 25 & data$weight <= 60, '30-60',
                         ifelse(data$weight > 60 & data$weight <= 85, '60-85', '85+'))
data <- data[, !(colnames(data) %in% 'weight')]

data$height_group <- ifelse(data$height >= 130 & data$height <= 160, '130-160',
                            ifelse(data$height > 160 & data$height <= 185, '160-185', '185+'))
data <- data[, !(colnames(data) %in% 'height')]


