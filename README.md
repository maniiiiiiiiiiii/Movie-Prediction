#  Movie Rating Prediction   

## **Project Overview**  
This project aims to build a predictive model for estimating movie ratings based on various attributes like genre, director, and cast. The model is trained on the [IMDB India Movies Dataset](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies) and evaluated using standard regression metrics.

---

## **Table of Contents**  
- [Objective](#objective)  
- [Dataset](#dataset)  
- [Approach](#approach)  
- [Data Preprocessing](#data-preprocessing)  
- [Feature Engineering](#feature-engineering)  
- [Model Training & Evaluation](#model-training--evaluation)  
- [Results](#results)  
- [How to Run the Project](#how-to-run-the-project)  
- [Future Improvements](#future-improvements)  

---

## **Objective**  
The primary goal is to develop a machine learning model that accurately predicts movie ratings based on available metadata, improving recommendation systems and user experiences.

---

## **Dataset**  
The dataset used for training and testing can be downloaded from:  
[IMDB India Movies Dataset](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies)  

### **Dataset Features**  
- **Title**: Name of the movie  
- **Genre**: Movie genre (Action, Drama, etc.)  
- **Director**: Name of the director  
- **Cast**: List of main actors  
- **Year**: Release year  
- **Duration**: Movie duration in minutes  
- **Rating**: IMDb rating (Target Variable)  

---

## **Approach**  
1. **Data Collection & Exploration**  
   - Load and analyze dataset structure.  
   - Identify and handle missing values.  

2. **Data Preprocessing**  
   - Encoding categorical variables.  
   - Filling missing values with appropriate methods.  

3. **Feature Engineering**  
   - Compute **director success rate** (average rating of movies by the same director).  
   - Compute **average rating of similar genres** to capture trends.  

4. **Model Training & Selection**  
   - Train multiple regression models:  
     - Linear Regression  
     - Random Forest Regressor  
     - Gradient Boosting Regressor  
   - Select the best-performing model based on evaluation metrics.  

5. **Model Evaluation**  
   - Evaluate using **MAE, RMSE, and R² score**.  

---

## **Data Preprocessing**  
### **Handling Missing Values**  
- Numerical missing values were filled with the **median**.  
- Categorical missing values were handled using **mode imputation** or dropped if necessary.  

### **Encoding Categorical Variables**  
- Used **Label Encoding** for categorical variables (`genre`, `director`, `cast`).  

---

## **Feature Engineering**  
To improve model accuracy, we engineered additional features:  
- **Director Success Rate**: The average rating of movies directed by the same person.  
- **Genre Average Rating**: The mean rating of all movies in the same genre.  

```python
df['director_success_rate'] = df.groupby('director')['rating'].transform('mean')
df['genre_avg_rating'] = df.groupby('genre')['rating'].transform('mean')
```

---

## **Model Training & Evaluation**  
### **Model Used: Random Forest Regressor**  
Random Forest was chosen for its ability to handle categorical variables and its robustness against overfitting.

```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### **Evaluation Metrics**  
- **Mean Absolute Error (MAE)**: Measures average prediction error.  
- **Root Mean Squared Error (RMSE)**: Penalizes larger errors.  
- **R² Score**: Measures how well the model explains variance in ratings.  

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}, RMSE: {rmse}, R²: {r2}")
```

---

## **Results**  
| Model  | MAE  | RMSE  | R² Score |
|--------|------|------|---------|
| Random Forest | 0.57 | 0.83 | 0.79 |

The **Random Forest Regressor** performed the best, achieving **0.79 R² score**, meaning it explains 79% of the variance in movie ratings.

---

## **How to Run the Project**  
### **Requirements**  
Ensure you have the following installed:  
- Python (3.7+)  
- Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- Jupyter Notebook (optional)  

### **Steps**  
1. **Clone the repository**  
   ```sh
   git clone https://github.com/maniiiiiiiiiiiii/movie-prediction.git
   cd movie-rating-prediction
   ```

2. **Install dependencies**  
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the script**  
   ```sh
   python movie_rating_prediction.py
   ```

4. **View results**  
   - Model performance metrics will be displayed in the terminal.  
   - The trained model is saved as `movie_rating_predictor.pkl`.  

---

## **Future Improvements**  
- **Hyperparameter Tuning**: Optimize Random Forest parameters for better accuracy.  
- **Deep Learning Model**: Try neural networks for better feature extraction.  
- **Additional Features**: Include **box office earnings**, **critic reviews**, and **user reviews sentiment**.  

---

## **Contributors**  
- **Manikanta**  
- [GitHub Profile](https://github.com/maniiiiiiiiiiiii)  


---

### **GitHub Repository Structure**  
```
📂 movie-rating-prediction  
│── 📄 README.md  # This Report  
│── 📄 movie_rating_prediction.py  # Main Python Script  
│── 📄 requirements.txt  # Dependencies  
│── 📄 imdb_india_movies.csv  # Dataset (or link in README)  
│── 📄 movie_rating_predictor.pkl  # Trained Model  
```

---
