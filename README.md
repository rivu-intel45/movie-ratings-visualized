# 🎬 Movie Ratings Visualized

A visual deep dive into how movies were rated in 2015 across **Fandango**, **Rotten Tomatoes**, **Metacritic**, and **IMDB**. Explore whether Fandango’s star ratings really tell the whole story, or if there’s hidden bias behind the numbers.

---

## 📊 What This Project Does

- Analyzes Fandango’s 2015 movie ratings and vote counts.
- Extracts movie release years from titles and visualizes how many movies appear per year.
- Compares Fandango scores with critics and users on Rotten Tomatoes, Metacritic, and IMDB. 
- Uses plots and correlations to reveal patterns, outliers, and rating discrepancies.  

---

## 📁 Project Contents

- `Fandangomoviereview.ipynb` – Main notebook: cleaning, EDA, plots, and correlation analysis.  
- `fandango_scrape.csv` – Raw Fandango movie ratings and votes. 
- `all_sites_scores.csv` – Combined ratings from multiple review sites.  
- `hotel_data_pandas.py` – Extra pandas practice script (not required for the main story).

---

## 🛠 Tech Stack
- Python
- pandas
- numpy
- matplotlib
- seaborn

- Install everything with:
- pip install -r requirements.txt
  ---

## 🚀 Getting Started

Clone the repo
git clone https://github.com/rivu-intel45/movie-ratings-visualized.git cd movie-ratings-visualized

Run the notebook
jupyter notebook Fandangomoviereview.ipynb

Open the notebook and run the cells top to bottom to reproduce all charts and statistics.

---

## 💡 Ideas to Extend

- Test statistically whether Fandango systematically rates higher than critics. 
- Add dashboards (Streamlit/Plotly) for interactive exploration. 
- Pull newer data and see if rating patterns changed after 2015.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork this repository  
2. Create a new branch: `git checkout -b feature/your-feature-name`  
3. Commit your changes: `git commit -m "Add your feature"`  
4. Push to the branch: `git push origin feature/your-feature-name`  
5. Open a Pull Request describing what you changed and why  

Please keep code clean, add comments where helpful, and update the notebook/README if your change affects the analysis or visuals.

---
## 📬 Contact

If you have questions, suggestions, or want to collaborate, feel free to reach out:
  
- Email: `protyaysahawork@gmail.com`  

You can also open an issue in this repository, and responses will be added there as soon as possible.

