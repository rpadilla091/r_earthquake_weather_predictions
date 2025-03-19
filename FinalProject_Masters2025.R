#Installing all necesary packages
#install.packages(c("ggplot2", "dplyr", "lubridate", "caret", "randomForest", "leaflet"))

# Load required libraries
library(ggplot2)      # For data visualization
library(dplyr)        # For data manipulation
library(lubridate)    # For date/time conversion
library(caret)        # For machine learning
library(randomForest) # For classification
library(leaflet)      # For interactive maps

# ------------------- DATA LOADING & CLEANING -------------------

# Load earthquake data
earthquake_df <- read.csv("earthquakeQuery.csv")

# Load weather data
weather_df <- read.csv("ImperialWeather.csv")

# Convert datetime columns to appropriate format
earthquake_df$datetime <- as.POSIXct(earthquake_df$time, format="%Y-%m-%dT%H:%M:%OSZ", tz="UTC")
weather_df$datetime <- as.Date(weather_df$DATE, format="%m/%d/%Y")

# Filter weather data from 2020 onward
weather_df <- filter(weather_df, datetime >= as.Date("2020-01-01"))

# Select relevant columns
weather_df <- weather_df %>%
  select(datetime, PRCP, TMAX, TMIN) %>%
  rename(max_temp = TMAX, min_temp = TMIN)

# Convert temperature from tenths of degrees Celsius to Fahrenheit
weather_df$max_temp <- (weather_df$max_temp / 10) * (9/5) + 32
weather_df$min_temp <- (weather_df$min_temp / 10) * (9/5) + 32

# Convert precipitation to binary (Presence: 1, Absence: 0)
weather_df$precipitation_present <- ifelse(is.na(weather_df$PRCP), 0, 1)

# Convert PRCP to numeric and normalize
weather_df$precipitation_percentage <- as.numeric(weather_df$PRCP)
weather_df$precipitation_percentage[is.na(weather_df$precipitation_percentage)] <- 0

# Select relevant earthquake columns
earthquake_df <- earthquake_df %>%
  select(datetime, latitude, longitude, depth, mag) %>%
  rename(magnitude = mag)

# Extract only date for merging
earthquake_df$date <- as.Date(earthquake_df$datetime)
weather_df$date <- as.Date(weather_df$datetime)

# Merge datasets by date
merged_df <- merge(earthquake_df, weather_df, by="date")

# ------------------- DATA VISUALIZATION -------------------

# 1. Time Series Plot of Earthquake Magnitude Over Time
ggplot(merged_df, aes(x = date, y = magnitude)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  labs(title = "Earthquake Magnitude Over Time", x = "Date", y = "Magnitude") +
  theme_minimal()

# 2. Scatter Plot - Magnitude vs Max Temperature
ggplot(merged_df, aes(x = max_temp, y = magnitude)) +
  geom_point(color = "blue", alpha = 0.5) +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Earthquake Magnitude vs Max Temperature", x = "Max Temperature (°F)", y = "Magnitude") +
  theme_minimal()

# 3. Boxplot of Earthquake Depths by Month
ggplot(merged_df, aes(x = factor(month(date)), y = depth)) +
  geom_boxplot(fill = "skyblue", color = "black") +
  labs(title = "Distribution of Earthquake Depths by Month", x = "Month", y = "Depth (km)") +
  theme_minimal()

# 4. Histogram of Earthquake Magnitudes
ggplot(merged_df, aes(x = magnitude)) +
  geom_histogram(fill = "green", bins = 10, color = "black") +
  labs(title = "Histogram of Earthquake Magnitudes", x = "Magnitude", y = "Frequency") +
  theme_minimal()

# 5. Interactive Map of Earthquake Locations
leaflet(merged_df) %>%
  addTiles() %>%
  addCircleMarkers(~longitude, ~latitude, radius = ~magnitude * 2, 
                   color = "red", fillOpacity = 0.6,
                   popup = ~paste("Magnitude:", magnitude)) %>%
  addLegend("bottomright", colors = "red", labels = "Earthquake Locations", title = "Legend")

# ------------------- MODELING -------------------

# Create binary classification target (Magnitude >= 4.5 as "High Magnitude")
merged_df$high_magnitude <- ifelse(merged_df$magnitude >= 4.5, "High", "Low")

# Define features and target variables
features <- c("max_temp", "min_temp", "precipitation_present", "precipitation_percentage")
X <- merged_df[, features]
y_reg <- merged_df$magnitude  # Regression target
y_clf <- as.factor(merged_df$high_magnitude)  # Classification target

# Split data into training and testing sets (80% train, 20% test)
set.seed(123)
trainIndex <- createDataPartition(y_clf, p = 0.8, list = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_reg_train <- y_reg[trainIndex]
y_reg_test <- y_reg[-trainIndex]
y_clf_train <- y_clf[trainIndex]
y_clf_test <- y_clf[-trainIndex]

# --------------- Regression Model (Predicting Magnitude) ---------------
lm_model <- lm(magnitude ~ max_temp + min_temp + precipitation_present + precipitation_percentage, 
               data = merged_df[trainIndex, ])
summary(lm_model)

# Predict on test set
y_reg_pred <- predict(lm_model, newdata = merged_df[-trainIndex, ])

# Calculate RMSE
rmse <- sqrt(mean((y_reg_test - y_reg_pred)^2))
print(paste("Regression RMSE:", round(rmse, 3)))

# --------------- Classification Model (Predicting High/Low Magnitude) ---------------

# Ensure that the classification target variable is treated as a factor
merged_df$high_magnitude <- as.factor(merged_df$high_magnitude)

rf_model <- randomForest(high_magnitude ~ max_temp + min_temp + precipitation_present + precipitation_percentage, 
                         data = merged_df[trainIndex, ], ntree = 100)


rf_pred <- predict(rf_model, newdata = merged_df[-trainIndex, ])

# Classification Accuracy
conf_matrix <- confusionMatrix(rf_pred, y_clf_test)
print(paste("Classification Accuracy:", round(conf_matrix$overall["Accuracy"], 3)))

# Feature Importance Plot
importance <- varImpPlot(rf_model, main = "Feature Importance (Random Forest)")

# ------------------- RESULTS & CONCLUSION -------------------
print("Modeling complete. Key findings:")
print(paste("Regression RMSE:", round(rmse, 3)))
print(conf_matrix)
