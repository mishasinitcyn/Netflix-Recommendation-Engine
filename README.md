# Netflix Movie Recommendation System
by *Mikhail Sinitcyn, Nicholas Zhang*

This repository implements a movie recommendation system using the Netflix Prize dataset enhanced with IMDb metadata. By combining collaborative filtering with engineered user preference features, the system predicts user movie ratings with 76% ROC AUC accuracy and provides interpretable insights into viewer preferences.

Full methodology and detailed findings available in [`report.pdf`](report.pdf).

## Overview
### Performance Metrics

- 76% ROC AUC Score with XGBoost classifier
- 69% precision on high ratings (4-5 stars)
- 60% recall on low ratings (1-3 stars)
- 64% overall F1 score

### Key Components

1. **Feature Engineering**
   - Mapped 10,000 movies to IMDb IDs
   - Extracted 450 features with 20+ occurrences
   - Engineered user preference features for top 10,000 users
   - Created high/low rating ratios per feature

2. **Model Implementation**
   - K-Means clustering by runtime and genre
   - DBSCAN outlier detection for unusual casting
   - XGBoost classification with random search hyperparameter tuning
   - LOF outlier detection for feature combinations

3. **Feature Selection**
   - Genre features (Thriller, Comedy, Family) highest predictors
   - 1980s most highly rated decade
   - 90-120min most preferred runtime
   - Removed 386 low-impact features without performance degradation



## Data Preprocessing
Merged Netflix Prize ratings with IMDb data to map movies to their features (cast, composers, directors, producers, genre, release year). Removed post-2005 films and TV shows. Mapped 10,000 movies to IMDb IDs to extract features, keeping only those with 20+ entries (450 features). Removed Country (noise from multiple release countries) and isAdult (only 20/9000 movies). Created user preference features for top 10,000 users as high(4-5) and low(1-3) rating ratios per feature.

## Exploratory Data Analysis
### Netflix Analysis
- Most reviewed movies have more high ratings than low
- Users with most ratings show more low ratings than high
- Top rated: Forrest Gump, Sixth Sense, Pirates of the Caribbean, Matrix
- Poorly rated: Wild Wild West, MIB II, Planet of the Apes
- Recent films (2002-2005) show increased negative rating ratio

### IMDb Analysis
- Most common features: 90-120min runtime, Drama, Comedy, 1990s/2000s
- Frequent actors: John Wayne, Gene Hackman, Michael Caine, De Niro
- Notable directors: Hitchcock, Woody Allen, Eastwood (30+ movies each)
- Strong correlations: Action-Adventure, Horror-Sci-Fi, Crime-Action
- Negative correlations: Comedy-Runtime>120min, Romance-Horror

## Clustering
### Movies
**K-Means**:
- 4 clusters optimal via Silhouette Score
- Primary clustering by Runtime, secondary by Genre
- Distinct groups: War/Drama>120min, Romance/Drama 60-90min, Action/Adventure>120min
- PCA reduction to 50 components showed similar Runtime/Genre hierarchy

**DBSCAN**:
- Functions as outlier detector due to sparse one-hot encoding
- Identifies feature-rich "superstar" movies as outliers (Batman, Antz, Hook)
- Less interpretable results with PCA-reduced features

### Users
- Preferences follow elliptical Gaussian distribution
- K-Means (k=2) shows trivial separation
- DBSCAN identifies extreme outliers (e.g., dedicated 60s action fans)

## Outlier Detection
**LOF**: Identifies unusual feature combinations, especially surprising cast ensembles (Royal Tenenbaums, Indecent Proposal, Mars Attacks)

**Isolation Forest**: Detected 462 outliers, mostly feature-rich movies with cross-genre "superstar" casts (Lawrence of Arabia, Predator 2, Batman Returns)

## Classification
Training on 100,000 ratings from 30 random users:

**K Nearest Neighbors**:
- High ratings: 65% precision, 57% recall
- Combined: 61% F1, 72% ROC 

**Logistic Regression**:
- High ratings: 66% precision, 59% recall
- Combined: 62% F1, 73% ROC AUC

**XGBoost**:
- High ratings: 69% precision, 60% recall
- Combined: 64% F1, 76% ROC AUC
- Hyperparameter tuning showed minimal improvement
- Removing the least significant 386 features, does not reduce the predictive abilities

## Feature Selection & Model Interpretation
- Genre (Thriller, Comedy, Family) highest predictors
- 1980s most highly rated decade
- 90-120min most preferred runtime
- Danny Elfman highest individual predictor
- Many actor features showed negligible weights

## Implementation
Notebooks available in respective folders:
- `ETL/`: Data cleaning, feature engineering
- `EDA/`: Netflix & IMDb analysis
- `Clustering/`: K-Means & DBSCAN implementations
- `Outliers/`: LOF & Isolation Forest detection
- `Classification/`: Logistic Regression & XGBoost models

For complete methodology and detailed findings, refer to `Report.pdf`.



## Setup
### Environment
- Create a virtual environment: `python3 -m venv venv`
- Activate the virtual environment: `source venv/bin/activate`
- Install the required libraries: `pip3 install -r requirements.txt`

### Data
- Download the [Netflix Prize Dataset](https://academictorrents.com/details/9b13183dc4d60676b773c9e2cd6de5e5542cee9a) and place all contents into `/data/netflix_prize`
- Download the [IMDB Datasets](https://datasets.imdbws.com/) and place all contents (7 gz files) into `/data/imdb`
